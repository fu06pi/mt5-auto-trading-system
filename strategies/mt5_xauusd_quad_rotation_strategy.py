"""XAUUSD Quad Rotation — 四重輪動策略.

John Kurisaki's quadruple stochastic rotation system.

Signal 1: Oversold divergence (counter-trend)
- All 4 Stochastics below 20 → price bounces → price new low + K1 divergence → enter

Signal 2: Flag pullback (trend-following)
- Price above 50EMA, pullback to 20/50EMA, K1 < 20, K4 > 85 → enter
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pymt5linux import MetaTrader5

sys.path.insert(0, "/home/chain4655/Documents/Projects/MT5")
from shared.account_metrics import AccountMetricsStore


LOGGER = logging.getLogger("xauusd_quad_rotation")
STATE_PATH_DEFAULT = "/home/chain4655/Documents/Projects/MT5/auto_quant/state/quad_rotation_state.json"
LOG_PATH_DEFAULT = "/home/chain4655/Documents/Projects/MT5/auto_quant/logs/quad_rotation.log"


@dataclasses.dataclass(frozen=True)
class StrategyConfig:
    symbol: str
    timeframe: str
    host: str
    port: int
    live: bool
    risk_pct: float
    stop_atr: float
    reward_multiple: float
    max_lots: float
    max_leverage: float
    max_drawdown_pct: float
    max_daily_loss_pct: float
    max_consecutive_losses: int
    cooldown_minutes: int
    loop_seconds: int
    lookback_bars: int
    atr_period: int
    ema_fast: int
    ema_mid: int
    ema_slow: int
    stoch_k1: int
    stoch_d1: int
    stoch_k2: int
    stoch_d2: int
    stoch_k3: int
    stoch_d3: int
    stoch_k4: int
    stoch_d4: int
    oversold_threshold: float
    overbought_threshold: float
    macro_strong_threshold: float
    pullback_max_deviation_atr: float
    state_path: str
    log_file: str
    terminal_path: Optional[str]
    deviation: int
    magic: int
    log_level: str


@dataclasses.dataclass
class StochValues:
    k: float
    d: float


@dataclasses.dataclass
class MarketSnapshot:
    bar_time: dt.datetime
    close: float
    high: float
    low: float
    atr: float
    stoch1: StochValues
    stoch2: StochValues
    stoch3: StochValues
    stoch4: StochValues
    ema_fast: float
    ema_mid: float
    ema_slow: float
    spread_points: float
    signal: str
    signal_type: str
    signal_price: float


@dataclasses.dataclass
class PositionState:
    ticket: int
    symbol: str
    magic: int = 0
    type: int = 0
    volume: float = 0.0
    price_open: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    profit: float = 0.0
    time_open: Optional[int] = None


@dataclasses.dataclass
class StrategyState:
    current_day: Optional[str] = None
    day_start_equity: Optional[float] = None
    last_equity: Optional[float] = None
    initial_equity: Optional[float] = None
    max_equity_seen: Optional[float] = None
    trades_today: int = 0
    consecutive_losses: int = 0
    last_trade_time: Optional[str] = None
    paused_reason: str = ""
    paused: bool = False


class QuadRotationStrategy:
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.mt5 = MetaTrader5(host=config.host, port=config.port)
        self.symbol = config.symbol
        self.timeframe = self._resolve_timeframe(config.timeframe)
        self.state_path = Path(config.state_path)
        self.log_file = Path(config.log_file)
        self.state = self._load_state()
        self.account_metrics = AccountMetricsStore()
        self._last_bar_time: Optional[dt.datetime] = None
        self._connect_fail_streak = 0
        self._last_reconnect_ts = 0.0
        self._initialized = False
        self._price_buffer: List[float] = []
        self._high_buffer: List[float] = []
        self._low_buffer: List[float] = []

    def _resolve_timeframe(self, tf: str) -> int:
        mapping = {
            "M1": self.mt5.TIMEFRAME_M1,
            "M2": self.mt5.TIMEFRAME_M2,
            "M3": self.mt5.TIMEFRAME_M3,
            "M5": self.mt5.TIMEFRAME_M5,
            "M15": self.mt5.TIMEFRAME_M15,
            "M30": self.mt5.TIMEFRAME_M30,
            "H1": self.mt5.TIMEFRAME_H1,
            "H4": self.mt5.TIMEFRAME_H4,
            "D1": self.mt5.TIMEFRAME_D1,
        }
        key = tf.upper()
        if key in mapping:
            return mapping[key]
        raise ValueError(f"Unsupported timeframe: {tf}")

    def run(self) -> None:
        self._connect()
        self._prepare_symbol()

        while True:
            try:
                snapshot = self._build_snapshot()
                if not self._initialized:
                    self._ensure_day_context(snapshot.bar_time)
                    self._seed_equity()
                    self._log("QuadRotation started: %s %s live=%s MACRO_THRESH=%.0f",
                              self.symbol, self.config.timeframe, self.config.live,
                              self.config.macro_strong_threshold)
                    self._initialized = True

                self._ensure_day_context(snapshot.bar_time)
                self._risk_guard(snapshot)

                if self._last_bar_time is not None and snapshot.bar_time <= self._last_bar_time:
                    time.sleep(self.config.loop_seconds)
                    continue

                self._last_bar_time = snapshot.bar_time
                self._handle_bar(snapshot)
            except KeyboardInterrupt:
                self._log("Interrupted by user, shutting down.")
                break
            except SystemExit:
                raise
            except Exception as exc:
                if self._is_transient_ipc_issue(exc):
                    self._handle_transient_ipc(exc)
                    continue
                self._log("Main loop error: %s", exc)
                time.sleep(self.config.loop_seconds)

        self._shutdown()

    def _load_state(self) -> StrategyState:
        if not self.state_path.exists():
            return StrategyState()
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return StrategyState(
                current_day=data.get("current_day"),
                day_start_equity=data.get("day_start_equity"),
                last_equity=data.get("last_equity"),
                initial_equity=data.get("initial_equity"),
                max_equity_seen=data.get("max_equity_seen"),
                trades_today=int(data.get("trades_today", 0)),
                consecutive_losses=int(data.get("consecutive_losses", 0)),
                last_trade_time=data.get("last_trade_time"),
                paused_reason=str(data.get("paused_reason", "")),
                paused=bool(data.get("paused", False)),
            )
        except Exception as exc:
            self._log("State load failed, starting fresh: %s", exc)
            return StrategyState()

    def _save_state(self) -> None:
        payload = {
            "current_day": self.state.current_day,
            "day_start_equity": self.state.day_start_equity,
            "last_equity": self.state.last_equity,
            "initial_equity": self.state.initial_equity,
            "max_equity_seen": self.state.max_equity_seen,
            "trades_today": self.state.trades_today,
            "consecutive_losses": self.state.consecutive_losses,
            "last_trade_time": self.state.last_trade_time,
            "paused_reason": self.state.paused_reason,
            "paused": self.state.paused,
        }
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except Exception as exc:
            self._log("State save failed: %s", exc)

    def _log(self, message: str, *args: Any) -> None:
        line = message % args if args else message
        LOGGER.info(line)
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(f"{dt.datetime.now().isoformat(sep=' ', timespec='seconds')} {line}\n")
        except Exception:
            pass

    def _connect(self) -> None:
        if self.config.terminal_path:
            self._log("Initializing terminal path: %s", self.config.terminal_path)
            ok = self.mt5.initialize(path=self.config.terminal_path)
        else:
            self._log("Initializing MT5 bridge without explicit terminal path")
            ok = self.mt5.initialize()
        self._log("initialize() -> %s | last_error=%s", ok, self.mt5.last_error())
        if not ok:
            raise RuntimeError(f"MT5 initialize failed: {self.mt5.last_error()}")

    def _prepare_symbol(self) -> None:
        if not self.mt5.symbol_select(self.symbol, True):
            raise RuntimeError(f"symbol_select failed: {self.mt5.last_error()}")
        info = self.mt5.symbol_info(self.symbol)
        if info is None:
            raise RuntimeError(f"symbol_info is None for {self.symbol}")
        self._log("Symbol ready: %s | digits=%s point=%s volume_min=%s volume_step=%s",
                  self.symbol, getattr(info, "digits", "?"), getattr(info, "point", "?"),
                  getattr(info, "volume_min", "?"), getattr(info, "volume_step", "?"))

    def _shutdown(self) -> None:
        try:
            self.mt5.shutdown()
        except Exception:
            pass

    def _seed_equity(self) -> None:
        equity = self._get_equity()
        metrics = self.account_metrics.update(equity, self.state.current_day)
        self.state.current_day = metrics.current_day
        self.state.initial_equity = metrics.initial_equity
        self.state.day_start_equity = metrics.day_start_equity
        self.state.max_equity_seen = metrics.max_equity_seen
        self.state.last_equity = metrics.equity
        self._save_state()

    def _get_equity(self) -> float:
        for attempt in range(3):
            info = self.mt5.account_info()
            if info is not None:
                equity = getattr(info, "equity", None)
                if equity is None:
                    equity = getattr(info, "balance", None)
                if equity is not None:
                    return float(equity)
            self._log("account_info unavailable (attempt %s/3): %s", attempt + 1, self.mt5.last_error())
            time.sleep(1)
        raise RuntimeError(f"account_info is None: {self.mt5.last_error()}")

    def _get_balance(self) -> float:
        info = self.mt5.account_info()
        if info is None:
            return 0.0
        return float(getattr(info, "balance", 0.0))

    def _ensure_day_context(self, bar_time: dt.datetime) -> None:
        day_key = bar_time.date().isoformat()
        if self.state.current_day is None:
            metrics = self.account_metrics.update(self._get_equity(), day_key)
            self.state.current_day = metrics.current_day
            self.state.day_start_equity = metrics.day_start_equity
            self.state.initial_equity = metrics.initial_equity
            self.state.max_equity_seen = metrics.max_equity_seen
            self.state.last_equity = metrics.equity
            self._save_state()
            return
        if self.state.current_day == day_key:
            return
        self.state.current_day = day_key
        metrics = self.account_metrics.update(self._get_equity(), day_key)
        self.state.day_start_equity = metrics.day_start_equity
        self.state.initial_equity = metrics.initial_equity
        self.state.max_equity_seen = metrics.max_equity_seen
        self.state.last_equity = metrics.equity
        self.state.trades_today = 0
        self.state.consecutive_losses = 0
        self.state.paused = False
        self.state.paused_reason = ""
        self._save_state()
        self._log("New trading day: %s | shared_day_start_equity=%.2f", day_key, metrics.day_start_equity)

    def _risk_guard(self, snapshot: MarketSnapshot) -> None:
        equity = self._get_equity()
        metrics = self.account_metrics.update(equity, snapshot.bar_time.date().isoformat())
        self.state.current_day = metrics.current_day
        self.state.last_equity = metrics.equity
        self.state.initial_equity = metrics.initial_equity
        self.state.day_start_equity = metrics.day_start_equity
        self.state.max_equity_seen = metrics.max_equity_seen

        day_dd = metrics.daily_dd
        total_dd = metrics.total_dd
        profit_pct = metrics.profit_pct

        self._log("Risk | equity=%.2f shared_day_start=%.2f shared_initial=%.2f shared_peak=%.2f day_dd=%.2f%% total_dd=%.2f%% profit=%.2f%% paused=%s",
                  metrics.equity, metrics.day_start_equity, metrics.initial_equity,
                  metrics.max_equity_seen, day_dd * 100, total_dd * 100,
                  profit_pct * 100, self.state.paused)
        self._save_state()

        if day_dd >= self.config.max_daily_loss_pct:
            self.state.paused = True
            self.state.paused_reason = f"Daily DD limit: {day_dd * 100:.2f}%"
            self._save_state()
            return

        if total_dd >= self.config.max_drawdown_pct:
            self.state.paused = True
            self.state.paused_reason = f"Total DD limit: {total_dd * 100:.2f}%"
            self._save_state()
            return

    def _is_transient_ipc_issue(self, exc: Exception) -> bool:
        text = str(exc).lower()
        keywords = ("broken pipe", "connection reset", "transport", "ipc", "timeout", "dead object", "10004", "no ipc")
        return any(keyword in text for keyword in keywords)

    def _handle_transient_ipc(self, exc: Exception) -> None:
        self._connect_fail_streak += 1
        self._log("Transient IPC issue (streak=%s): %s", self._connect_fail_streak, exc)
        now = time.time()
        if now - self._last_reconnect_ts < 5:
            time.sleep(2)
            return
        self._last_reconnect_ts = now
        try:
            self.mt5.shutdown()
        except Exception:
            pass
        time.sleep(1)
        try:
            self._connect()
            self._prepare_symbol()
            self._connect_fail_streak = 0
        except Exception as reconnect_exc:
            self._log("Transient IPC reconnect failed; will retry next loop: %s", reconnect_exc)
            time.sleep(self.config.loop_seconds)

    def _fetch_bars(self, min_count: int) -> List[Any]:
        bars = self.mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, min_count)
        if bars is None or len(bars) == 0:
            self._log("copy_rates_from_pos unavailable (fetched %s bars)", 0 if bars is None else len(bars))
            raise RuntimeError(f"copy_rates_from_pos returned None: {self.mt5.last_error()}")
        return list(bars)

    def _compute_sma(self, values: Sequence[float], period: int) -> float:
        if len(values) < period or period <= 0:
            return 0.0
        return sum(values[-period:]) / period

    def _compute_ema(self, values: Sequence[float], period: int) -> float:
        if len(values) < period or period <= 0:
            return 0.0
        alpha = 2.0 / (period + 1)
        ema = sum(values[:period]) / period
        for v in values[period:]:
            ema = v * alpha + ema * (1 - alpha)
        return ema

    def _compute_stochastic(self, highs: Sequence[float], lows: Sequence[float], closes: Sequence[float],
                            k_period: int, d_period: int) -> Tuple[List[float], List[float]]:
        raw_k: List[float] = []
        for i in range(len(closes)):
            if i < k_period - 1:
                raw_k.append(50.0)
                continue
            hh = max(highs[i - k_period + 1:i + 1])
            ll = min(lows[i - k_period + 1:i + 1])
            if hh == ll:
                raw_k.append(raw_k[-1] if raw_k else 50.0)
            else:
                raw_k.append((closes[i] - ll) / (hh - ll) * 100.0)

        k_values: List[float] = list(raw_k)
        d_values: List[float] = []
        for i in range(len(k_values)):
            if i < d_period - 1:
                d_values.append(50.0)
            else:
                d_values.append(sum(k_values[i - d_period + 1:i + 1]) / d_period)
        return k_values, d_values

    def _compute_atr(self, highs: Sequence[float], lows: Sequence[float], closes: Sequence[float],
                     period: int) -> List[float]:
        tr_values: List[float] = []
        for i in range(len(closes)):
            if i == 0:
                tr_values.append(highs[i] - lows[i])
            else:
                tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
                tr_values.append(tr)
        atr_values: List[float] = []
        for i in range(len(tr_values)):
            if i < period:
                atr_values.append(sum(tr_values[:i + 1]) / (i + 1))
            else:
                atr_values.append((atr_values[-1] * (period - 1) + tr_values[i]) / period)
        return atr_values

    def _build_snapshot(self) -> MarketSnapshot:
        need_bars = self.config.lookback_bars + self.config.atr_period + max(
            self.config.stoch_k1, self.config.stoch_k2, self.config.stoch_k3, self.config.stoch_k4,
            self.config.ema_slow
        ) * 2
        bars = self._fetch_bars(need_bars)

        closes = [float(b[4]) for b in bars]
        highs = [float(b[2]) for b in bars]
        lows = [float(b[3]) for b in bars]
        times = [dt.datetime.fromtimestamp(int(b[0]), tz=dt.timezone.utc) for b in bars]

        latest = bars[-1]
        bar_time = times[-1]
        close = closes[-1]
        high = highs[-1]
        low = lows[-1]

        atr_values = self._compute_atr(highs, lows, closes, self.config.atr_period)
        atr = atr_values[-1] if atr_values else 0.0

        k1, d1 = self._compute_stochastic(highs, lows, closes, self.config.stoch_k1, self.config.stoch_d1)
        k2, d2 = self._compute_stochastic(highs, lows, closes, self.config.stoch_k2, self.config.stoch_d2)
        k3, d3 = self._compute_stochastic(highs, lows, closes, self.config.stoch_k3, self.config.stoch_d3)
        k4, d4 = self._compute_stochastic(highs, lows, closes, self.config.stoch_k4, self.config.stoch_d4)

        ema_fast = self._compute_ema(closes, self.config.ema_fast)
        ema_mid = self._compute_ema(closes, self.config.ema_mid)
        ema_slow = self._compute_ema(closes, self.config.ema_slow)

        spread = 0.0
        try:
            tick = self.mt5.symbol_info_tick(self.symbol)
            if tick is not None:
                ask = float(getattr(tick, "ask", 0.0))
                bid = float(getattr(tick, "bid", 0.0))
                if ask > 0 and bid > 0:
                    spread = (ask - bid) / self._point() if self._point() > 0 else 0.0
        except Exception:
            pass

        signal = "NONE"
        signal_type = ""
        signal_price = 0.0

        signal, signal_type, signal_price = self._check_signals(
            closes, highs, lows, k1, d1, k2, d2, k3, d3, k4, d4,
            ema_fast, ema_mid, ema_slow, atr, bar_time
        )

        stoch1 = StochValues(k=k1[-1], d=d1[-1])
        stoch2 = StochValues(k=k2[-1], d=d2[-1])
        stoch3 = StochValues(k=k3[-1], d=d3[-1])
        stoch4 = StochValues(k=k4[-1], d=d4[-1])

        return MarketSnapshot(
            bar_time=bar_time, close=close, high=high, low=low,
            atr=atr, stoch1=stoch1, stoch2=stoch2, stoch3=stoch3, stoch4=stoch4,
            ema_fast=ema_fast, ema_mid=ema_mid, ema_slow=ema_slow,
            spread_points=spread, signal=signal, signal_type=signal_type,
            signal_price=signal_price,
        )

    def _check_signals(self, closes, highs, lows,
                        k1, d1, k2, d2, k3, d3, k4, d4,
                        ema_fast, ema_mid, ema_slow, atr, bar_time):
        n = len(closes)
        oversold = self.config.oversold_threshold
        overbought = self.config.overbought_threshold
        macro = self.config.macro_strong_threshold

        idx = n - 1
        current_close = closes[-1]
        current_high = highs[-1]
        current_low = lows[-1]

        s1_sell, s1_buy = self._signal1_check(closes, highs, lows, k1, d1, k2, d2, k3, d3, k4, d4)
        s2_buy, s2_sell = self._signal2_check(closes, highs, lows, k1, d1, k2, d2, k3, d3, k4, d4,
                                               ema_fast, ema_mid, ema_slow, atr)

        if s1_buy:
            return "BUY", "S1", current_close
        if s2_buy:
            return "BUY", "S2", current_close
        if s1_sell:
            return "SELL", "S1", current_close
        if s2_sell:
            return "SELL", "S2", current_close
        return "NONE", "", 0.0

    def _signal1_check(self, closes, highs, lows, k1, d1, k2, d2, k3, d3, k4, d4):
        n = len(closes)
        oversold = self.config.oversold_threshold
        overbought = self.config.overbought_threshold
        lookback = min(40, n - 20)

        buy = self._detect_signal1_buy(closes, highs, lows, k1, k2, k3, k4, oversold, lookback)
        sell = self._detect_signal1_sell(closes, highs, lows, k1, k2, k3, k4, overbought, lookback)
        return sell, buy

    def _detect_signal1_buy(self, closes, highs, lows, k1, k2, k3, k4, oversold, lookback):
        n = len(closes)
        scan_start = n - lookback

        for i in range(scan_start, n - 3):
            if not (k1[i] < oversold and k2[i] < oversold and k3[i] < oversold and k4[i] < oversold):
                continue
            oversold_idx = i
            oversold_low = lows[i]
            oversold_k1 = k1[i]

            bounce_high = oversold_low
            bounce_idx = oversold_idx
            for j in range(oversold_idx + 1, n - 1):
                if highs[j] > bounce_high:
                    bounce_high = highs[j]
                    bounce_idx = j

            if bounce_idx <= oversold_idx:
                continue

            if closes[bounce_idx] <= closes[oversold_idx]:
                continue

            new_low_idx = -1
            new_low = bounce_high
            for j in range(bounce_idx + 1, n):
                if lows[j] < lows[oversold_idx]:
                    new_low = lows[j]
                    new_low_idx = j
                    break

            if new_low_idx < 0:
                continue

            if k1[new_low_idx] > k1[oversold_idx] and k1[new_low_idx] < oversold:
                return True
        return False

    def _detect_signal1_sell(self, closes, highs, lows, k1, k2, k3, k4, overbought, lookback):
        n = len(closes)
        scan_start = n - lookback

        for i in range(scan_start, n - 3):
            if not (k1[i] > overbought and k2[i] > overbought and k3[i] > overbought and k4[i] > overbought):
                continue
            overbought_idx = i
            overbought_high = highs[i]
            overbought_k1 = k1[i]

            dip_low = overbought_high
            dip_idx = overbought_idx
            for j in range(overbought_idx + 1, n - 1):
                if lows[j] < dip_low:
                    dip_low = lows[j]
                    dip_idx = j

            if dip_idx <= overbought_idx:
                continue

            if closes[dip_idx] >= closes[overbought_idx]:
                continue

            new_high_idx = -1
            for j in range(dip_idx + 1, n):
                if highs[j] > highs[overbought_idx]:
                    new_high_idx = j
                    break

            if new_high_idx < 0:
                continue

            if k1[new_high_idx] < k1[overbought_idx] and k1[new_high_idx] > overbought:
                return True
        return False

    def _signal2_check(self, closes, highs, lows, k1, d1, k2, d2, k3, d3, k4, d4,
                        ema_fast, ema_mid, ema_slow, atr):
        n = len(closes)
        oversold = self.config.oversold_threshold
        overbought = self.config.overbought_threshold
        macro = self.config.macro_strong_threshold

        buy = self._detect_signal2_buy(closes, highs, lows, k1, k4,
                                       ema_fast, ema_mid, ema_slow, atr,
                                       oversold, macro)
        sell = self._detect_signal2_sell(closes, highs, lows, k1, k4,
                                         ema_fast, ema_mid, ema_slow, atr,
                                         overbought, macro)
        return buy, sell

    def _detect_signal2_buy(self, closes, highs, lows, k1, k4,
                             ema_fast, ema_mid, ema_slow, atr,
                             oversold, macro):
        idx = len(closes) - 1
        price = closes[idx]

        if price < ema_mid:
            return False

        if k1[idx] >= oversold:
            return False

        if k4[idx] < macro:
            return False

        pullback_max = atr * self.config.pullback_max_deviation_atr
        ema_diff = price - ema_fast
        if ema_diff > pullback_max or ema_diff < -pullback_max * 2:
            near_ema20 = abs(price - ema_fast) < pullback_max * 1.5
            near_ema50 = abs(price - ema_mid) < pullback_max * 0.8
            if not (near_ema20 or near_ema50):
                return False

        return True

    def _detect_signal2_sell(self, closes, highs, lows, k1, k4,
                              ema_fast, ema_mid, ema_slow, atr,
                              overbought, macro):
        idx = len(closes) - 1
        price = closes[idx]

        if price > ema_mid:
            return False

        if k1[idx] <= overbought:
            return False

        if k4[idx] > 100 - macro:
            pass

        pullback_max = atr * self.config.pullback_max_deviation_atr
        ema_diff = ema_fast - price
        if ema_diff > pullback_max or ema_diff < -pullback_max * 2:
            near_ema20 = abs(price - ema_fast) < pullback_max * 1.5
            near_ema50 = abs(price - ema_mid) < pullback_max * 0.8
            if not (near_ema20 or near_ema50):
                return False

        return True

    def _handle_bar(self, snapshot: MarketSnapshot) -> None:
        self._sync_closed_trades()
        positions = self._positions()
        foreign_positions = self._foreign_positions()

        self._log(
            "Bar %s | close=%.2f atr=%.2f K1=%.1f D1=%.1f K2=%.1f K3=%.1f K4=%.1f "
            "EMA20=%.1f EMA50=%.1f EMA200=%.1f signal=%s type=%s spread=%.0f "
            "positions=%d foreign=%d trades_today=%d losses=%d",
            snapshot.bar_time, snapshot.close, snapshot.atr,
            snapshot.stoch1.k, snapshot.stoch1.d,
            snapshot.stoch2.k, snapshot.stoch3.k, snapshot.stoch4.k,
            snapshot.ema_fast, snapshot.ema_mid, snapshot.ema_slow,
            snapshot.signal, snapshot.signal_type,
            snapshot.spread_points,
            len(positions), len(foreign_positions),
            self.state.trades_today, self.state.consecutive_losses,
        )

        if self.state.paused:
            self._log("Trading paused: %s", self.state.paused_reason)
            return

        if self.state.consecutive_losses >= self.config.max_consecutive_losses:
            self._log("Consecutive loss cap reached: %d", self.state.consecutive_losses)
            return

        if not positions:
            if foreign_positions:
                self._log("Foreign position present (%d); skip new entry.", len(foreign_positions))
                return
            if snapshot.signal != "NONE":
                self._enter(snapshot)
            return

        pos = positions[0]
        self._manage_position(snapshot, pos)

    def _enter(self, snapshot: MarketSnapshot) -> None:
        direction = snapshot.signal
        if direction not in ("BUY", "SELL"):
            return

        ask, bid = self._tick_prices()
        price = ask if direction == "BUY" else bid

        sl, tp = self._build_sl_tp(direction, price, snapshot.atr)

        volume = self._size_position(direction, price, sl)

        request = self._order_request(direction, volume, price, sl, tp)
        result = self._send_order_with_filling_fallback(request)

        if result is not None:
            retcode = getattr(result, "retcode", -1)
            ticket = getattr(result, "order", 0)
            if retcode in {self.mt5.TRADE_RETCODE_DONE, self.mt5.TRADE_RETCODE_PLACED}:
                self.state.trades_today += 1
                self._save_state()
                self._log("ENTRY %s: ticket=%d volume=%.2f price=%.2f sl=%.2f tp=%.2f | type=%s",
                          direction, ticket, volume, price, sl, tp, snapshot.signal_type)
            else:
                self._log("ENTRY FAILED %s: retcode=%d ticket=%d comment=%s",
                          direction, retcode, ticket, getattr(result, "comment", ""))
        else:
            self._log("ENTRY FAILED %s: order_send returned None err=%s",
                      direction, self.mt5.last_error())

    def _manage_position(self, snapshot: MarketSnapshot, pos: PositionState) -> None:
        ask, bid = self._tick_prices()
        current_price = bid if pos.type == self.mt5.POSITION_TYPE_BUY else ask
        open_price = pos.price_open
        direction = "BUY" if pos.type == self.mt5.POSITION_TYPE_BUY else "SELL"
        profit = (current_price - open_price) if direction == "BUY" else (open_price - current_price)

        if profit >= snapshot.atr * 0.5:
            new_sl = open_price if direction == "BUY" else open_price
            if direction == "BUY":
                new_sl = max(pos.sl, open_price)
            else:
                new_sl = min(pos.sl, open_price) if pos.sl > 0 else open_price
            if pos.sl <= 0 or (direction == "BUY" and new_sl > pos.sl) or (direction == "SELL" and (pos.sl <= 0 or new_sl < pos.sl)):
                self._modify_position(pos, sl=new_sl, tp=pos.tp)

    def _sync_closed_trades(self) -> None:
        try:
            history = self.mt5.history_deals_get(
                int(time.time()) - 86400 * 7, int(time.time()) + 3600
            )
            if history is None:
                return
            for deal in history:
                deal_magic = int(getattr(deal, "magic", 0))
                if deal_magic != self.config.magic:
                    continue
                deal_entry = getattr(deal, "entry", 0)
                if deal_entry != 1:
                    continue
                deal_time = str(getattr(deal, "time", 0))
                if deal_time == self.state.last_trade_time:
                    continue
                profit = float(getattr(deal, "profit", 0.0))
                if profit < 0:
                    self.state.consecutive_losses += 1
                else:
                    self.state.consecutive_losses = 0
                self.state.last_trade_time = deal_time
                self._save_state()
        except Exception as exc:
            self._log("Sync closed trades error: %s", exc)

    def _build_sl_tp(self, direction: str, price: float, atr: float) -> Tuple[float, float]:
        sl_distance = atr * self.config.stop_atr
        tp_distance = sl_distance * self.config.reward_multiple
        point = self._point()

        if direction == "BUY":
            sl = price - sl_distance
            tp = price + tp_distance
        else:
            sl = price + sl_distance
            tp = price - tp_distance

        digits = self._digits()
        sl = self._round_to_digits(sl, digits)
        tp = self._round_to_digits(tp, digits)
        if abs(price - sl) < point:
            sl = price - point if direction == "BUY" else price + point
        if abs(price - tp) < point:
            tp = price + point if direction == "BUY" else price - point
        return sl, tp

    def _size_position(self, direction: str, price: float, sl: float) -> float:
        equity = self._get_equity()
        risk_amount = equity * self.config.risk_pct
        info = self.mt5.symbol_info(self.symbol)
        if info is None:
            raise RuntimeError("symbol_info unavailable")

        risk_per_lot = self._risk_per_lot(direction, price, sl)
        if risk_per_lot <= 0:
            raise RuntimeError("risk_per_lot invalid")

        raw_volume = risk_amount / risk_per_lot
        raw_volume = min(raw_volume, self.config.max_lots)

        margin_lot = self._margin_per_lot(direction, price)
        if margin_lot > 0:
            acc = self.mt5.account_info()
            free_margin = float(getattr(acc, "margin_free", equity) if acc is not None else equity)
            max_by_margin = (free_margin * 0.85) / margin_lot
            raw_volume = min(raw_volume, max_by_margin)

        min_vol = float(getattr(info, "volume_min", 0.01))
        step = float(getattr(info, "volume_step", 0.01))
        raw_volume = max(raw_volume, min_vol)
        volume = self._round_volume(raw_volume, step)
        return min(volume, self.config.max_lots)

    def _risk_per_lot(self, direction: str, price: float, sl: float) -> float:
        order_type = self.mt5.ORDER_TYPE_BUY if direction == "BUY" else self.mt5.ORDER_TYPE_SELL
        try:
            value = self.mt5.order_calc_profit(order_type, self.symbol, 1.0, price, sl)
            if value is not None:
                return abs(float(value))
        except Exception:
            pass
        info = self.mt5.symbol_info(self.symbol)
        if info is None:
            return 0.0
        contract_size = float(getattr(info, "trade_contract_size", 1.0))
        return abs(price - sl) * contract_size

    def _margin_per_lot(self, direction: str, price: float) -> float:
        order_type = self.mt5.ORDER_TYPE_BUY if direction == "BUY" else self.mt5.ORDER_TYPE_SELL
        try:
            value = self.mt5.order_calc_margin(order_type, self.symbol, 1.0, price)
            if value is not None:
                return abs(float(value))
        except Exception:
            pass
        return 0.0

    def _order_request(self, direction: str, volume: float, price: float, sl: float, tp: float) -> Dict[str, Any]:
        filling = self._select_filling_mode()
        return {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(volume),
            "type": self.mt5.ORDER_TYPE_BUY if direction == "BUY" else self.mt5.ORDER_TYPE_SELL,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": int(self.config.deviation),
            "magic": int(self.config.magic),
            "comment": "quad-rotation" if self.config.live else "quad-dryrun",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }

    def _select_filling_mode(self) -> int:
        info = self.mt5.symbol_info(self.symbol)
        candidates: List[int] = []
        if info is not None:
            raw = getattr(info, "filling_mode", None)
            if raw is None:
                raw = getattr(info, "trade_fill_mode", None)
            if raw is not None:
                mapping = {0: self.mt5.ORDER_FILLING_FOK, 1: self.mt5.ORDER_FILLING_IOC,
                           2: self.mt5.ORDER_FILLING_RETURN, 3: self.mt5.ORDER_FILLING_BOC}
                try:
                    raw_i = int(raw)
                    if raw_i in mapping:
                        candidates.append(mapping[raw_i])
                except Exception:
                    pass
        for mode in (self.mt5.ORDER_FILLING_IOC, self.mt5.ORDER_FILLING_RETURN, self.mt5.ORDER_FILLING_FOK):
            if mode not in candidates:
                candidates.append(mode)
        return candidates[0]

    def _send_order_with_filling_fallback(self, request: Dict[str, Any]) -> Any:
        requested = int(request.get("type_filling", self.mt5.ORDER_FILLING_RETURN))
        fallback_modes = [requested, self.mt5.ORDER_FILLING_RETURN, self.mt5.ORDER_FILLING_IOC, self.mt5.ORDER_FILLING_FOK]
        tried: List[int] = []
        last_result = None
        for mode in fallback_modes:
            if mode in tried:
                continue
            tried.append(mode)
            req = dict(request)
            req["type_filling"] = mode
            result = self.mt5.order_send(req)
            last_result = result
            self._log("order_send(type_filling=%s) -> retcode=%s", mode,
                      getattr(result, "retcode", "?") if result is not None else "None")
            if result is not None:
                code = getattr(result, "retcode", None)
                if code in {self.mt5.TRADE_RETCODE_DONE, self.mt5.TRADE_RETCODE_PLACED, self.mt5.TRADE_RETCODE_DONE_PARTIAL}:
                    return result
                if code not in {self.mt5.TRADE_RETCODE_INVALID_FILL, self.mt5.TRADE_RETCODE_INVALID_ORDER}:
                    return result
        return last_result

    def _modify_position(self, pos: PositionState, sl: float, tp: float) -> None:
        request = {
            "action": self.mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": pos.ticket,
            "sl": float(sl),
            "tp": float(tp),
            "magic": int(self.config.magic),
            "comment": "quad-mgt",
        }
        if self.config.live:
            result = self.mt5.order_send(request)
            self._log("SLTP modify result: retcode=%s", getattr(result, "retcode", "?") if result is not None else "None")
        else:
            self._log("DRY-RUN SLTP modify: sl=%.2f tp=%.2f", sl, tp)

    def close_all_positions(self) -> None:
        positions = self._positions()
        if not positions:
            return
        ask, bid = self._tick_prices()
        for pos in positions:
            price = bid if pos.type == self.mt5.POSITION_TYPE_BUY else ask
            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": self.mt5.ORDER_TYPE_SELL if pos.type == self.mt5.POSITION_TYPE_BUY else self.mt5.ORDER_TYPE_BUY,
                "position": pos.ticket,
                "price": price,
                "deviation": int(self.config.deviation),
                "magic": int(self.config.magic),
                "comment": "quad-close-all",
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": self._select_filling_mode(),
            }
            if self.config.live:
                result = self._send_order_with_filling_fallback(request)
                self._log("Close result: retcode=%s", getattr(result, "retcode", "?") if result is not None else "None")
            else:
                self._log("DRY-RUN close request: %s", request)

    def _all_positions(self) -> List[PositionState]:
        raw = self.mt5.positions_get(symbol=self.symbol)
        if not raw:
            return []
        out: List[PositionState] = []
        for pos in raw:
            out.append(PositionState(
                ticket=int(getattr(pos, "ticket", 0)),
                symbol=str(getattr(pos, "symbol", self.symbol)),
                magic=int(getattr(pos, "magic", 0) or 0),
                type=int(getattr(pos, "type", 0)),
                volume=float(getattr(pos, "volume", 0.0)),
                price_open=float(getattr(pos, "price_open", 0.0)),
                sl=float(getattr(pos, "sl", 0.0)),
                tp=float(getattr(pos, "tp", 0.0)),
                profit=float(getattr(pos, "profit", 0.0)),
                time_open=int(getattr(pos, "time", 0)) if getattr(pos, "time", None) is not None else None,
            ))
        return out

    def _foreign_positions(self) -> List[PositionState]:
        return [pos for pos in self._all_positions() if int(getattr(pos, "magic", 0)) != int(self.config.magic)]

    def _positions(self) -> List[PositionState]:
        return [pos for pos in self._all_positions() if int(getattr(pos, "magic", 0)) == int(self.config.magic)]

    def _tick_prices(self) -> Tuple[float, float]:
        tick = self.mt5.symbol_info_tick(self.symbol)
        if tick is None:
            raise RuntimeError(f"symbol_info_tick failed: {self.mt5.last_error()}")
        ask = float(getattr(tick, "ask", 0.0) or getattr(tick, "last", 0.0) or 0.0)
        bid = float(getattr(tick, "bid", 0.0) or getattr(tick, "last", 0.0) or 0.0)
        if ask <= 0 or bid <= 0:
            raise RuntimeError(f"Invalid bid/ask: ask={ask} bid={bid}")
        return ask, bid

    def _point(self) -> float:
        info = self.mt5.symbol_info(self.symbol)
        if info is None:
            raise RuntimeError("symbol_info unavailable")
        return float(getattr(info, "point", 0.0) or 0.0)

    def _digits(self) -> int:
        info = self.mt5.symbol_info(self.symbol)
        if info is None:
            return 2
        return int(getattr(info, "digits", 2) or 2)

    def _round_to_digits(self, value: float, digits: int) -> float:
        return round(value, digits)

    def _round_volume(self, volume: float, step: float) -> float:
        return round(volume / step) * step


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="XAUUSD Quad Rotation — 四重輪動")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M15")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18812)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--risk-pct", type=float, default=0.008)
    parser.add_argument("--stop-atr", type=float, default=1.5)
    parser.add_argument("--reward-multiple", type=float, default=1.5)
    parser.add_argument("--max-lots", type=float, default=3.0)
    parser.add_argument("--max-leverage", type=float, default=10.0)
    parser.add_argument("--max-drawdown-pct", type=float, default=0.05)
    parser.add_argument("--max-daily-loss-pct", type=float, default=0.025)
    parser.add_argument("--max-consecutive-losses", type=int, default=5)
    parser.add_argument("--cooldown-minutes", type=int, default=30)
    parser.add_argument("--loop-seconds", type=int, default=10)
    parser.add_argument("--lookback-bars", type=int, default=200)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--ema-fast", type=int, default=20)
    parser.add_argument("--ema-mid", type=int, default=50)
    parser.add_argument("--ema-slow", type=int, default=200)
    parser.add_argument("--stoch-k1", type=int, default=9)
    parser.add_argument("--stoch-d1", type=int, default=3)
    parser.add_argument("--stoch-k2", type=int, default=14)
    parser.add_argument("--stoch-d2", type=int, default=3)
    parser.add_argument("--stoch-k3", type=int, default=40)
    parser.add_argument("--stoch-d3", type=int, default=4)
    parser.add_argument("--stoch-k4", type=int, default=60)
    parser.add_argument("--stoch-d4", type=int, default=1)
    parser.add_argument("--oversold-threshold", type=float, default=20.0)
    parser.add_argument("--overbought-threshold", type=float, default=80.0)
    parser.add_argument("--macro-strong-threshold", type=float, default=85.0)
    parser.add_argument("--pullback-max-deviation-atr", type=float, default=0.8)
    parser.add_argument("--state-path", default=STATE_PATH_DEFAULT)
    parser.add_argument("--log-file", default=LOG_PATH_DEFAULT)
    parser.add_argument("--terminal-path",
                        default=r"C:\Program Files\MetaTrader 5\terminal64.exe")
    parser.add_argument("--deviation", type=int, default=30)
    parser.add_argument("--magic", type=int, default=210513)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    terminal_path = args.terminal_path.strip() or None
    config = StrategyConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        host=args.host,
        port=args.port,
        live=bool(args.live),
        risk_pct=float(args.risk_pct),
        stop_atr=float(args.stop_atr),
        reward_multiple=float(args.reward_multiple),
        max_lots=float(args.max_lots),
        max_leverage=float(args.max_leverage),
        max_drawdown_pct=float(args.max_drawdown_pct),
        max_daily_loss_pct=float(args.max_daily_loss_pct),
        max_consecutive_losses=int(args.max_consecutive_losses),
        cooldown_minutes=int(args.cooldown_minutes),
        loop_seconds=int(args.loop_seconds),
        lookback_bars=int(args.lookback_bars),
        atr_period=int(args.atr_period),
        ema_fast=int(args.ema_fast),
        ema_mid=int(args.ema_mid),
        ema_slow=int(args.ema_slow),
        stoch_k1=int(args.stoch_k1),
        stoch_d1=int(args.stoch_d1),
        stoch_k2=int(args.stoch_k2),
        stoch_d2=int(args.stoch_d2),
        stoch_k3=int(args.stoch_k3),
        stoch_d3=int(args.stoch_d3),
        stoch_k4=int(args.stoch_k4),
        stoch_d4=int(args.stoch_d4),
        oversold_threshold=float(args.oversold_threshold),
        overbought_threshold=float(args.overbought_threshold),
        macro_strong_threshold=float(args.macro_strong_threshold),
        pullback_max_deviation_atr=float(args.pullback_max_deviation_atr),
        state_path=args.state_path,
        log_file=args.log_file,
        terminal_path=terminal_path,
        deviation=int(args.deviation),
        magic=int(args.magic),
        log_level=args.log_level,
    )

    strategy = QuadRotationStrategy(config)
    strategy.run()


if __name__ == "__main__":
    main()
