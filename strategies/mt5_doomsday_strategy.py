"""MT5 Python strategy inspired by the user's "doomsday" article.

Important:
- Defaults to demo-safe / dry-run mode.
- Uses current equity for compounding.
- Enforces hard caps on lot size, leverage, and account drawdown.
- Designed to run through pymt5linux on Linux while controlling a Windows MT5 terminal.

Run example:
    source ~/Documents/Sample/Python/venv313/bin/activate
    python /home/chain4655/Documents/Projects/MT5/strategies/mt5_doomsday_strategy.py --symbol XAUUSD --timeframe M5 --host 127.0.0.1 --port 18812 --live
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import logging
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pymt5linux import MetaTrader5


LOGGER = logging.getLogger("mt5_doomsday_strategy")
FILE_LOGGER = logging.getLogger("mt5_doomsday_strategy.file")
LOG_FILE_PATH = "/home/chain4655/Documents/Projects/MT5/mt5_doomsday_strategy_supervised.log"


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
    tp_min_usd: float
    tp_max_usd: float
    long_bias: float
    trend_threshold: float
    roll_trigger_pct: float
    cooldown_minutes: int
    max_leverage: float
    max_drawdown_pct: float
    max_daily_loss_pct: float
    max_lots: float
    magic: int
    deviation: int
    loop_seconds: int
    lookback_bars: int
    atr_period: int
    fast_sma: int
    slow_sma: int
    high_vol_only: bool
    high_vol_atr_pct: float
    high_vol_range_atr: float
    high_vol_breakout_lookback: int
    high_vol_min_momentum: float
    high_vol_spike_atr: float
    high_vol_min_breakout_atr: float
    high_vol_min_close_location: float
    log_level: str
    terminal_path: Optional[str]
    log_file: Optional[str]
    state_path: Optional[str]




@dataclasses.dataclass
class MarketSnapshot:
    bar_time: dt.datetime
    close: float
    high: float
    low: float
    atr: float
    fast_sma: float
    slow_sma: float
    momentum: float
    score: float
    signal: str
    atr_pct: float
    range_atr: float
    spike_atr: float
    breakout_atr: float
    close_location: float
    high_volatility: bool


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

class DoomsdayMT5Strategy:
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.mt5 = MetaTrader5(host=config.host, port=config.port)
        self.symbol = config.symbol
        self.timeframe = self._resolve_timeframe(config.timeframe)
        self._last_bar_time: Optional[dt.datetime] = None
        self._initial_equity: Optional[float] = None
        self._session_start_equity: Optional[float] = None
        self._max_equity_seen: Optional[float] = None
        self._last_signal: str = "NONE"
        self._last_trade_time: Optional[dt.datetime] = None
        self._last_trade_close_time: Optional[dt.datetime] = None
        self._pending_reverse_signal: str = "NONE"
        self._pending_reverse_since: Optional[dt.datetime] = None
        self._symbol_point: float = 0.01
        self._symbol_digits: int = 2
        self._symbol_volume_min: float = 0.01
        self._symbol_volume_step: float = 0.01
        self._symbol_volume_max: float = 1.0
        self._symbol_contract_size: float = 1.0
        self._log_file_path: str = config.log_file or LOG_FILE_PATH
        self._state_path: Optional[str] = config.state_path
        self._equity_cache: Optional[float] = None
        self._equity_cache_at: float = 0.0
        self._connected_once: bool = False
        self._reconnect_cooldown_until: float = 0.0
        self._positions_cache: List[PositionState] = []
        self._positions_cache_at: float = 0.0
        self._last_account_warning_at: float = 0.0

    def run(self) -> None:
        self._connect()
        self._prepare_symbol()
        self._seed_equity()

        self._log_file("Strategy started: %s %s live=%s", self.symbol, self.config.timeframe, self.config.live)
        self._write_state({
            "status": "started",
            "symbol": self.symbol,
            "timeframe": self.config.timeframe,
            "live": self.config.live,
            "pid": os.getpid(),
        })
        LOGGER.info("Strategy started: %s %s live=%s", self.symbol, self.config.timeframe, self.config.live)

        while True:
            try:
                self._risk_guard()
                snapshot = self._build_snapshot()
                if self._last_bar_time is not None and snapshot.bar_time <= self._last_bar_time:
                    time.sleep(self.config.loop_seconds)
                    continue

                self._last_bar_time = snapshot.bar_time
                self._handle_bar(snapshot)
            except KeyboardInterrupt:
                LOGGER.info("Interrupted by user, shutting down.")
                break
            except Exception as exc:
                LOGGER.exception("Main loop error: %s", exc)
                time.sleep(self.config.loop_seconds)

        self._shutdown()

    def _log_file(self, message: str, *args: Any) -> None:
        try:
            with open(self._log_file_path, "a", encoding="utf-8") as f:
                f.write((message % args) + "\n")
        except OSError as exc:
            LOGGER.warning("File log write failed: %s", exc)

    def _last_error_safe(self) -> Any:
        try:
            return self.mt5.last_error()
        except Exception as exc:
            return f"last_error unavailable: {exc}"

    def _write_state(self, payload: Dict[str, Any]) -> None:
        if not self._state_path:
            return
        try:
            with open(self._state_path, "w", encoding="utf-8") as f:
                import json
                json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        except OSError as exc:
            LOGGER.warning("State write failed: %s", exc)

    def _connect(self) -> None:
        last_error: Optional[Any] = None
        now = time.time()
        if self._connected_once and now < self._reconnect_cooldown_until:
            return
        for attempt in range(6):
            should_log_connect = not self._connected_once or attempt > 0
            if self.config.terminal_path:
                if should_log_connect:
                    LOGGER.info("Initializing terminal path: %s (attempt %s/6)", self.config.terminal_path, attempt + 1)
                    self._log_file("Initializing terminal path: %s (attempt %s/6)", self.config.terminal_path, attempt + 1)
                ok = self.mt5.initialize(path=self.config.terminal_path)
            else:
                if should_log_connect:
                    LOGGER.info("Initializing MT5 bridge without explicit terminal path (attempt %s/6)", attempt + 1)
                    self._log_file("Initializing MT5 bridge without explicit terminal path (attempt %s/6)", attempt + 1)
                ok = self.mt5.initialize()
            last_error = self._last_error_safe()
            if ok and self._is_connected():
                self._connected_once = True
                self._equity_cache_at = time.time()
                return
            self._reconnect_cooldown_until = time.time() + 300.0
            if attempt == 0 or attempt == 5:
                LOGGER.info("initialize() -> %s | last_error=%s", ok, last_error)
                self._log_file("initialize() -> %s | last_error=%s", ok, last_error)
            try:
                self.mt5.shutdown()
            except Exception as exc:
                LOGGER.warning("MT5 shutdown failed during connect: %s", exc)
            time.sleep(2 + attempt)
        raise RuntimeError(f"MT5 bridge not ready after retries: {last_error}")

    def _expected_login_and_server(self) -> Tuple[Optional[str], Optional[str]]:
        try:
            ini_path = Path("/home/chain4655/.mt5/drive_c/Program Files/MetaTrader 5/Config/common.ini")
            text = ini_path.read_text(encoding="utf-16-le", errors="ignore")
        except Exception:
            return None, None
        login = None
        server = None
        for line in text.splitlines():
            if line.startswith("Login="):
                value = line.split("=", 1)[1].strip()
                if value.isdigit():
                    login = value
            elif line.startswith("Server="):
                value = line.split("=", 1)[1].strip()
                if value and ":" not in value:
                    server = value
        return login, server

    def _is_connected(self) -> bool:
        try:
            ti = self.mt5.terminal_info()
            ai = self.mt5.account_info()
            if ti is None or ai is None:
                return False
            login, server = self._expected_login_and_server()
            if login is not None and str(getattr(ai, "login", "")) != login:
                return False
            if server is not None and str(getattr(ai, "server", "")) != server:
                return False
            return True
        except Exception:
            return False

    def _shutdown(self) -> None:
        try:
            self.mt5.shutdown()
        except Exception as exc:
            LOGGER.warning("MT5 shutdown failed: %s", exc)

    def _prepare_symbol(self) -> None:
        last_error: Any = None
        info: Any = None
        for attempt in range(3):
            selected = self.mt5.symbol_select(self.symbol, True)
            last_error = self._last_error_safe()
            if selected:
                info = self.mt5.symbol_info(self.symbol)
                if info is not None:
                    break
                last_error = self._last_error_safe()
            if attempt < 2:
                LOGGER.warning(
                    "Symbol info unavailable for %s (attempt %s/3): %s",
                    self.symbol,
                    attempt + 1,
                    last_error,
                )
                self._log_file(
                    "Symbol info unavailable for %s (attempt %s/3): %s",
                    self.symbol,
                    attempt + 1,
                    last_error,
                )
                try:
                    self.mt5.shutdown()
                except Exception:
                    pass
                self.mt5 = MetaTrader5(host=self.config.host, port=self.config.port)
                if self.config.terminal_path:
                    ok = self.mt5.initialize(path=self.config.terminal_path)
                else:
                    ok = self.mt5.initialize()
                if not ok:
                    last_error = self._last_error_safe()
                    self._connected_once = False
                else:
                    self._connected_once = True
                time.sleep(1.0 + attempt)
        if info is None:
            raise RuntimeError(f"symbol_info is None for {self.symbol}: {last_error}")

        self._symbol_digits = int(getattr(info, "digits", 2) or 2)
        self._symbol_point = float(getattr(info, "point", 0.01) or 0.01)
        self._symbol_volume_min = float(getattr(info, "volume_min", 0.01) or 0.01)
        self._symbol_volume_step = float(getattr(info, "volume_step", 0.01) or 0.01)
        self._symbol_volume_max = float(getattr(info, "volume_max", 1.0) or 1.0)
        self._symbol_contract_size = float(getattr(info, "trade_contract_size", 1.0) or 1.0)
        LOGGER.info(
            "Symbol ready: %s | digits=%s point=%s volume_min=%s volume_step=%s volume_max=%s",
            self.symbol,
            self._symbol_digits,
            self._symbol_point,
            self._symbol_volume_min,
            self._symbol_volume_step,
            self._symbol_volume_max,
        )

    def _seed_equity(self) -> None:
        equity = self._get_equity()
        self._initial_equity = equity
        self._session_start_equity = equity
        self._max_equity_seen = equity
        self._log_file("Equity seeded: %.2f", equity)
        LOGGER.info("Equity seeded: %.2f", equity)

    def _get_equity(self) -> float:
        now = time.time()
        if self._equity_cache is not None and (now - self._equity_cache_at) < 30.0:
            return self._equity_cache
        try:
            info = self.mt5.account_info()
        except EOFError as exc:
            info = None
            LOGGER.warning("account_info stream closed: %s", exc)
            self._connected_once = False
        except Exception as exc:
            info = None
            LOGGER.warning("account_info failed: %s", exc)
        if info is not None:
            equity = getattr(info, "equity", None)
            if equity is None:
                equity = getattr(info, "balance", None)
            if equity is not None:
                value = float(equity)
                if value <= 0:
                    raise RuntimeError(
                        f"account equity is non-positive ({value}); refusing to use invalid/unfunded login"
                    )
                self._equity_cache = value
                self._equity_cache_at = now
                return value
        if (now - self._last_account_warning_at) >= 120.0:
            LOGGER.warning("account_info unavailable; using cached equity: %s", self._last_error_safe())
            self._last_account_warning_at = now
        if self._equity_cache is not None:
            return self._equity_cache
        if now >= self._reconnect_cooldown_until:
            self._reconnect_cooldown_until = now + 300.0
            self._connected_once = False
            try:
                self._connect()
            except Exception as exc:
                LOGGER.warning("Reconnect attempt failed: %s", exc)
        if self._equity_cache is not None:
            return self._equity_cache
        raise RuntimeError(f"account_info is None after reconnect attempts: {self._last_error_safe()}")

    def _risk_guard(self) -> None:
        equity = self._get_equity()
        if self._max_equity_seen is None or equity > self._max_equity_seen:
            self._max_equity_seen = equity

        if self._initial_equity is None:
            self._initial_equity = equity
        if self._session_start_equity is None:
            self._session_start_equity = equity

        dd_from_peak = 1.0 - (equity / self._max_equity_seen) if self._max_equity_seen else 0.0
        dd_from_start = 1.0 - (equity / self._session_start_equity) if self._session_start_equity else 0.0

        if dd_from_peak >= self.config.max_drawdown_pct:
            LOGGER.warning(
                "Max drawdown hit: %.2f%% >= %.2f%%, closing all positions and stopping.",
                dd_from_peak * 100,
                self.config.max_drawdown_pct * 100,
            )
            self.close_all_positions()
            raise SystemExit(1)

        if dd_from_start >= self.config.max_daily_loss_pct:
            LOGGER.warning(
                "Daily loss hit: %.2f%% >= %.2f%%, closing all positions and stopping.",
                dd_from_start * 100,
                self.config.max_daily_loss_pct * 100,
            )
            self.close_all_positions()
            raise SystemExit(1)

    def _resolve_timeframe(self, tf: str) -> int:
        key = tf.strip().upper()
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
        if key not in mapping:
            raise ValueError(f"Unsupported timeframe: {tf}")
        return mapping[key]

    def _build_snapshot(self) -> MarketSnapshot:
        need = max(self.config.slow_sma, self.config.atr_period) + 5
        bars = self._fetch_bars(min_count=need)
        if len(bars) < need:
            raise RuntimeError(
                f"Not enough bars for signal calculation: got={len(bars)} need={need} lookback={self.config.lookback_bars}"
            )

        last_closed = bars[-2]
        closes = [bar["close"] for bar in bars[:-1]]
        highs = [bar["high"] for bar in bars[:-1]]
        lows = [bar["low"] for bar in bars[:-1]]

        atr = self._atr(bars[:-1], self.config.atr_period)
        fast_sma = statistics.fmean(closes[-self.config.fast_sma :])
        slow_sma = statistics.fmean(closes[-self.config.slow_sma :])
        momentum = (closes[-1] - closes[-4]) / atr if atr > 0 and len(closes) >= 4 else 0.0
        momentum = self._clamp(momentum, -2.0, 2.0)
        atr_pct = atr / max(float(last_closed["close"]), 1e-9)
        regime_lookback = min(20, len(highs), len(lows))
        range_atr = (max(highs[-regime_lookback:]) - min(lows[-regime_lookback:])) / max(atr, 1e-9)
        last_range = max(float(last_closed["high"]) - float(last_closed["low"]), 0.0)
        spike_atr = last_range / max(atr, 1e-9)
        close_location = (
            (float(last_closed["close"]) - float(last_closed["low"])) / last_range
            if last_range > 0
            else 0.5
        )
        breakout_atr = self._breakout_atr(closes, highs, lows, atr)
        high_volatility = self._is_high_volatility_regime(atr_pct, range_atr, spike_atr)

        score = self._score_signal(closes, highs, lows, atr, fast_sma, slow_sma, momentum)
        signal = self._decide_signal(score)
        if self.config.high_vol_only and not high_volatility:
            signal = "NONE"

        bar_time = self._bar_time(last_closed)
        return MarketSnapshot(
            bar_time=bar_time,
            close=float(last_closed["close"]),
            high=float(last_closed["high"]),
            low=float(last_closed["low"]),
            atr=atr,
            fast_sma=fast_sma,
            slow_sma=slow_sma,
            momentum=momentum,
            score=score,
            signal=signal,
            atr_pct=atr_pct,
            range_atr=range_atr,
            spike_atr=spike_atr,
            breakout_atr=breakout_atr,
            close_location=close_location,
            high_volatility=high_volatility,
        )

    def _refresh_market_data_client(self) -> None:
        """Refresh pymt5linux client before OHLC pulls to avoid Wine IPC flaps.

        On this host, a persistent pymt5linux client can keep account_info() working
        while copy_rates_from_pos() starts returning (-10004, 'No IPC connection').
        Recreating the client before market-data reads mirrors the hardened Momentum
        Surfer path and prevents the strategy from crashing into supervisor backoff.
        """
        try:
            self.mt5.shutdown()
        except Exception:
            pass
        self.mt5 = MetaTrader5(host=self.config.host, port=self.config.port)
        if self.config.terminal_path:
            ok = self.mt5.initialize(path=self.config.terminal_path)
        else:
            ok = self.mt5.initialize()
        if not ok:
            self._connected_once = False
            raise RuntimeError(f"MT5 initialize failed before market data fetch: {self._last_error_safe()}")
        self._connected_once = True
        selected = self.mt5.symbol_select(self.symbol, True)
        if not selected:
            raise RuntimeError(f"symbol_select failed before market data fetch: {self._last_error_safe()}")

    def _fetch_bars(self, min_count: Optional[int] = None) -> List[Dict[str, float]]:
        fetch_count = max(int(self.config.lookback_bars), int(min_count or 0))
        last_error: Any = None
        for attempt in range(3):
            try:
                self._refresh_market_data_client()
                rates = self.mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, fetch_count)
                if rates is not None:
                    bars: List[Dict[str, float]] = []
                    for row in list(rates):
                        bars.append(self._normalize_bar(row))
                    if bars:
                        return bars
                last_error = self._last_error_safe()
            except Exception as exc:
                last_error = exc
                self._connected_once = False
            if attempt < 2:
                LOGGER.warning(
                    "copy_rates_from_pos unavailable (attempt %s/3): %s",
                    attempt + 1,
                    last_error,
                )
                self._log_file(
                    "copy_rates_from_pos unavailable (attempt %s/3): %s",
                    attempt + 1,
                    last_error,
                )
                time.sleep(1.0 + attempt)
        raise RuntimeError(f"copy_rates_from_pos returned no bars after retries: {last_error}")

    def _normalize_bar(self, row: Any) -> Dict[str, float]:
        def get(key: str, fallback_index: Optional[int] = None) -> float:
            try:
                if isinstance(row, dict):
                    return float(row[key])
                try:
                    return float(row[key])
                except Exception:
                    pass
                if hasattr(row, key):
                    return float(getattr(row, key))
                if fallback_index is not None:
                    return float(row[fallback_index])
            except Exception:
                pass
            raise KeyError(key)

        ts = int(get("time", 0))
        return {
            "time": float(ts),
            "open": get("open", 1),
            "high": get("high", 2),
            "low": get("low", 3),
            "close": get("close", 4),
            "tick_volume": get("tick_volume", 5) if self._has_field(row, "tick_volume") else 0.0,
        }

    def _has_field(self, row: Any, name: str) -> bool:
        try:
            if isinstance(row, dict):
                return name in row
            if hasattr(row, "dtype") and getattr(row.dtype, "names", None):
                return name in row.dtype.names
            return hasattr(row, name)
        except Exception:
            return False

    def _bar_time(self, bar: Dict[str, float]) -> dt.datetime:
        return dt.datetime.fromtimestamp(int(bar["time"]))

    def _atr(self, bars: Sequence[Dict[str, float]], period: int) -> float:
        if len(bars) < period + 1:
            raise RuntimeError("Not enough bars for ATR")
        trs: List[float] = []
        for i in range(1, period + 1):
            curr = bars[-i]
            prev = bars[-i - 1]
            tr = max(
                curr["high"] - curr["low"],
                abs(curr["high"] - prev["close"]),
                abs(curr["low"] - prev["close"]),
            )
            trs.append(tr)
        atr = statistics.fmean(trs)
        return max(atr, self._point() * 5)

    def _score_signal(
        self,
        closes: Sequence[float],
        highs: Sequence[float],
        lows: Sequence[float],
        atr: float,
        fast_sma: float,
        slow_sma: float,
        momentum: float,
    ) -> float:
        last_close = closes[-1]
        breakout_lookback = max(5, min(self.config.high_vol_breakout_lookback, len(highs) - 1))
        recent_high = max(highs[-breakout_lookback - 1 : -1])
        recent_low = min(lows[-breakout_lookback - 1 : -1])

        trend = 0.0
        if last_close > fast_sma > slow_sma:
            trend = 0.40
        elif last_close < fast_sma < slow_sma:
            trend = -0.40
        else:
            if last_close > slow_sma:
                trend = 0.12
            elif last_close < slow_sma:
                trend = -0.12

        breakout = 0.0
        if atr > 0:
            up_break = self._clamp((last_close - recent_high) / atr, -1.5, 1.5)
            dn_break = self._clamp((recent_low - last_close) / atr, -1.5, 1.5)
            breakout = up_break * 0.45 + dn_break * -0.45

        momentum_component = self._clamp(momentum, -2.0, 2.0) * 0.45
        bias_component = (self.config.long_bias - 0.5) * 0.20

        score = trend + breakout + momentum_component + bias_component
        return self._clamp(score, -1.5, 1.5)

    def _breakout_atr(
        self,
        closes: Sequence[float],
        highs: Sequence[float],
        lows: Sequence[float],
        atr: float,
    ) -> float:
        if atr <= 0 or len(closes) < 2:
            return 0.0
        breakout_lookback = max(5, min(self.config.high_vol_breakout_lookback, len(highs) - 1))
        recent_high = max(highs[-breakout_lookback - 1 : -1])
        recent_low = min(lows[-breakout_lookback - 1 : -1])
        last_close = closes[-1]
        if last_close > recent_high:
            return (last_close - recent_high) / atr
        if last_close < recent_low:
            return (last_close - recent_low) / atr
        return 0.0

    def _is_high_volatility_regime(self, atr_pct: float, range_atr: float, spike_atr: float) -> bool:
        atr_ok = atr_pct >= self.config.high_vol_atr_pct
        expansion_ok = range_atr >= self.config.high_vol_range_atr or spike_atr >= self.config.high_vol_spike_atr
        return atr_ok and expansion_ok

    def _decide_signal(self, score: float) -> str:
        if score >= self.config.trend_threshold:
            return "BUY"
        if score <= -self.config.trend_threshold:
            return "SELL"
        return "NONE"

    def _high_vol_entry_ok(self, snapshot: MarketSnapshot) -> bool:
        if not snapshot.high_volatility:
            return False
        if abs(snapshot.momentum) < self.config.high_vol_min_momentum:
            return False
        if snapshot.atr <= 0 or snapshot.atr_pct <= 0:
            return False
        if abs(snapshot.score) < self.config.trend_threshold:
            return False
        if snapshot.signal == "BUY":
            if snapshot.momentum <= 0:
                return False
            if snapshot.breakout_atr < self.config.high_vol_min_breakout_atr:
                return False
            if snapshot.close_location < self.config.high_vol_min_close_location:
                return False
        elif snapshot.signal == "SELL":
            if snapshot.momentum >= 0:
                return False
            if snapshot.breakout_atr > -self.config.high_vol_min_breakout_atr:
                return False
            if snapshot.close_location > (1.0 - self.config.high_vol_min_close_location):
                return False
        else:
            return False
        return True

    def _handle_bar(self, snapshot: MarketSnapshot) -> None:
        all_positions = self._all_positions()
        positions = [
            pos for pos in all_positions
            if pos.symbol == self.symbol and int(getattr(pos, "magic", 0)) == int(self.config.magic)
        ]
        foreign_positions = [pos for pos in all_positions if pos.symbol != self.symbol]
        self._log_file(
            "Bar %s | close=%.2f atr=%.2f atr_pct=%.5f range_atr=%.2f spike_atr=%.2f breakout_atr=%.2f close_loc=%.2f high_vol=%s fast=%.2f slow=%.2f momentum=%.2f score=%.2f signal=%s positions=%d foreign=%d",
            snapshot.bar_time,
            snapshot.close,
            snapshot.atr,
            snapshot.atr_pct,
            snapshot.range_atr,
            snapshot.spike_atr,
            snapshot.breakout_atr,
            snapshot.close_location,
            snapshot.high_volatility,
            snapshot.fast_sma,
            snapshot.slow_sma,
            snapshot.momentum,
            snapshot.score,
            snapshot.signal,
            len(positions),
            len(foreign_positions),
        )
        LOGGER.info(
            "Bar %s | close=%.2f atr=%.2f atr_pct=%.5f range_atr=%.2f spike_atr=%.2f breakout_atr=%.2f close_loc=%.2f high_vol=%s fast=%.2f slow=%.2f momentum=%.2f score=%.2f signal=%s positions=%d",
            snapshot.bar_time,
            snapshot.close,
            snapshot.atr,
            snapshot.atr_pct,
            snapshot.range_atr,
            snapshot.spike_atr,
            snapshot.breakout_atr,
            snapshot.close_location,
            snapshot.high_volatility,
            snapshot.fast_sma,
            snapshot.slow_sma,
            snapshot.momentum,
            snapshot.score,
            snapshot.signal,
            len(positions),
        )

        if self.config.high_vol_only and not positions and not self._high_vol_entry_ok(snapshot):
            if snapshot.signal != "NONE":
                self._log_file(
                    "ENTRY_FILTERED high_vol_only signal=%s score=%.2f momentum=%.2f atr_pct=%.5f range_atr=%.2f spike_atr=%.2f breakout_atr=%.2f close_loc=%.2f high_vol=%s",
                    snapshot.signal,
                    snapshot.score,
                    snapshot.momentum,
                    snapshot.atr_pct,
                    snapshot.range_atr,
                    snapshot.spike_atr,
                    snapshot.breakout_atr,
                    snapshot.close_location,
                    snapshot.high_volatility,
                )
            return

        if not positions:
            if snapshot.signal != "NONE" and self._cooldown_ok(snapshot.bar_time):
                self._log_file("No open position; signal=%s", snapshot.signal)
                self._enter(snapshot)
            return

        pos = positions[0]
        if self._should_roll(snapshot, pos):
            self._log_file("Roll trigger met, closing and re-evaluating on next bar.")
            LOGGER.info("Roll trigger met, closing and re-evaluating on next bar.")
            self.close_all_positions()
            self._last_signal = "NONE"
            self._last_trade_close_time = snapshot.bar_time
            self._pending_reverse_signal = "NONE"
            self._pending_reverse_since = None
            return

        if self._should_reverse(snapshot, pos):
            now = snapshot.bar_time
            if self._pending_reverse_signal != snapshot.signal:
                self._pending_reverse_signal = snapshot.signal
                self._pending_reverse_since = now
                self._log_file("Reverse signal detected; waiting 15 minutes before reversing.")
                LOGGER.info("Reverse signal detected; waiting 15 minutes before reversing.")
                return

            if self._pending_reverse_since is not None:
                elapsed_minutes = (now - self._pending_reverse_since).total_seconds() / 60.0
                if elapsed_minutes < 15.0:
                    self._log_file(
                        "Reverse signal pending for %.1f minutes; need 15.0 minutes before reversing.",
                        elapsed_minutes,
                    )
                    LOGGER.info(
                        "Reverse signal pending for %.1f minutes; need 15.0 minutes before reversing.",
                        elapsed_minutes,
                    )
                    return

            self._log_file("Reverse signal persisted 15 minutes; closing and entering new direction.")
            LOGGER.info("Reverse signal persisted 15 minutes; closing and entering new direction.")
            self.close_all_positions()
            self._last_trade_close_time = snapshot.bar_time
            self._pending_reverse_signal = "NONE"
            self._pending_reverse_since = None
            time.sleep(1)
            if self._cooldown_ok(snapshot.bar_time):
                self._enter(snapshot)
            return

        self._pending_reverse_signal = "NONE"
        self._pending_reverse_since = None
        self._maybe_trail(snapshot, pos)

    def _all_positions(self) -> List[PositionState]:
        now = time.time()
        if (now - self._positions_cache_at) < 2.0:
            return list(self._positions_cache)
        try:
            raw = self.mt5.positions_get()
            if not raw:
                self._positions_cache = []
                self._positions_cache_at = now
                return []
            out: List[PositionState] = []
            for pos in raw:
                out.append(
                    PositionState(
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
                    )
                )
            self._positions_cache = out
            self._positions_cache_at = now
            return list(out)
        except EOFError as exc:
            LOGGER.warning("positions_get stream closed; reconnecting once: %s", exc)
            self._connected_once = False
            self._reconnect_cooldown_until = 0.0
            try:
                self._connect()
            except Exception as conn_exc:
                LOGGER.warning("Reconnect attempt failed while reading positions: %s", conn_exc)
            return list(self._positions_cache)
        except Exception as exc:
            LOGGER.warning("positions_get failed: %s", exc)
            return list(self._positions_cache)

    def _symbol_positions(self) -> List[PositionState]:
        return [pos for pos in self._all_positions() if pos.symbol == self.symbol]

    def _foreign_positions(self) -> List[PositionState]:
        return [pos for pos in self._all_positions() if pos.symbol != self.symbol]

    def _positions(self) -> List[PositionState]:
        return [pos for pos in self._symbol_positions() if int(getattr(pos, "magic", 0)) == int(self.config.magic)]

    def _should_roll(self, snapshot: MarketSnapshot, pos: PositionState) -> bool:
        if self._initial_equity is None:
            return False
        equity = self._get_equity()
        if equity <= 0:
            return False
        unrealized_profit_pct = pos.profit / max(equity, 1e-9)
        return unrealized_profit_pct >= self.config.roll_trigger_pct

    def _cooldown_ok(self, bar_time: dt.datetime) -> bool:
        cutoff = float(self.config.cooldown_minutes)
        if self._last_trade_time is not None:
            elapsed_entry = (bar_time - self._last_trade_time).total_seconds() / 60.0
            if elapsed_entry < cutoff:
                return False
        if self._last_trade_close_time is not None:
            elapsed_close = (bar_time - self._last_trade_close_time).total_seconds() / 60.0
            if elapsed_close < cutoff:
                return False
        return True

    def _should_reverse(self, snapshot: MarketSnapshot, pos: PositionState) -> bool:
        if self.config.high_vol_only and not self._high_vol_entry_ok(snapshot):
            return False
        if pos.type == self.mt5.POSITION_TYPE_BUY and snapshot.signal == "SELL":
            return True
        if pos.type == self.mt5.POSITION_TYPE_SELL and snapshot.signal == "BUY":
            return True
        return False

    def _maybe_trail(self, snapshot: MarketSnapshot, pos: PositionState) -> None:
        # Move stop to lock profits once the trade is comfortably ahead.
        if snapshot.atr <= 0:
            return
        ask, bid = self._tick_prices()
        current_price = bid if pos.type == self.mt5.POSITION_TYPE_BUY else ask
        profit_move = (current_price - pos.price_open) if pos.type == self.mt5.POSITION_TYPE_BUY else (pos.price_open - current_price)
        trail_trigger = snapshot.atr * (1.25 if snapshot.high_volatility else 2.0)
        lock_distance = snapshot.atr * (0.25 if snapshot.high_volatility else 0.5)
        if profit_move < trail_trigger:
            return

        if pos.type == self.mt5.POSITION_TYPE_BUY:
            new_sl = max(pos.sl, pos.price_open + lock_distance)
            if pos.sl <= 0 or new_sl > pos.sl:
                self._modify_position(pos, sl=new_sl, tp=pos.tp)
        else:
            new_sl = min(pos.sl if pos.sl > 0 else current_price + snapshot.atr * 100, pos.price_open - lock_distance)
            if pos.sl <= 0 or new_sl < pos.sl:
                self._modify_position(pos, sl=new_sl, tp=pos.tp)

    def _enter(self, snapshot: MarketSnapshot) -> None:
        direction = snapshot.signal
        if direction not in {"BUY", "SELL"}:
            return

        self._log_file(
            "ENTRY_ATTEMPT %s bar=%s score=%.2f atr=%.2f",
            direction,
            snapshot.bar_time,
            snapshot.score,
            snapshot.atr,
        )
        LOGGER.info(
            "ENTRY_ATTEMPT %s bar=%s score=%.2f atr=%.2f",
            direction,
            snapshot.bar_time,
            snapshot.score,
            snapshot.atr,
        )

        try:
            ask, bid = self._tick_prices()
            price = ask if direction == "BUY" else bid
            sl, tp = self._build_sl_tp(direction, price, snapshot.atr, snapshot.score)
            volume = self._size_position(direction, price, sl)
            if volume <= 0:
                raise RuntimeError("Calculated volume is zero")

            request = self._order_request(direction, volume, price, sl, tp)
            equity = self._get_equity()
            self._log_file(
                "ENTRY %s volume=%.2f price=%.2f sl=%.2f tp=%.2f equity=%.2f score=%.2f",
                direction,
                volume,
                price,
                sl,
                tp,
                equity,
                snapshot.score,
            )
            LOGGER.warning(
                "ENTRY %s volume=%.2f price=%.2f sl=%.2f tp=%.2f equity=%.2f score=%.2f",
                direction,
                volume,
                price,
                sl,
                tp,
                equity,
                snapshot.score,
            )

            if self.config.live:
                result = self._send_order_with_filling_fallback(request)
                result_dict = self._result_to_dict(result)
                self._log_file("order_send result: %s", result_dict)
                LOGGER.warning("order_send result: %s", result_dict)
                if not self._is_trade_success(result):
                    self._log_file("ENTRY_REJECTED %s result=%s last_error=%s", direction, result_dict, self._last_error_safe())
                    raise RuntimeError(f"order_send not successful: {result_dict} last_error={self._last_error_safe()}")
                self._log_file("order_send confirmed: %s", result_dict)
            else:
                self._log_file("DRY-RUN request: %s", request)
                LOGGER.info("DRY-RUN request: %s", request)

            self._last_signal = direction
            self._last_trade_time = snapshot.bar_time
            self._last_trade_close_time = None
        except Exception as exc:
            self._log_file("ENTRY_ERROR %s: %s", direction, exc)
            LOGGER.exception("ENTRY_ERROR %s: %s", direction, exc)
            raise

    def _build_sl_tp(self, direction: str, price: float, atr: float, score: float) -> Tuple[float, float]:
        sl_distance = atr * self.config.stop_atr
        point = self._point()
        min_tp_distance = max(self.config.tp_min_usd / max(1.0, self._contract_value_per_point()), point)
        max_tp_distance = max(self.config.tp_max_usd / max(1.0, self._contract_value_per_point()), min_tp_distance)
        strength = self._score_to_strength(score)
        tp_multiplier = self._interpolate(min_tp_distance, max_tp_distance, strength) / max(sl_distance, point)
        tp_distance = sl_distance * max(tp_multiplier, 0.1)

        if direction == "BUY":
            sl = price - sl_distance
            tp = price + tp_distance
        else:
            sl = price + sl_distance
            tp = price - tp_distance

        digits = self._digits()
        sl = self._round_to_digits(sl, digits)
        tp = self._round_to_digits(tp, digits)
        # Avoid zero-distance SL/TP.
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
            try:
                acc = self.mt5.account_info()
            except Exception as exc:
                acc = None
                self._log_file("ENTRY_WARN margin account_info unavailable: %s", exc)
                LOGGER.warning("ENTRY_WARN margin account_info unavailable: %s", exc)
            free_margin = float(getattr(acc, "margin_free", equity) if acc is not None else equity)
            max_by_margin = (free_margin * 0.85) / margin_lot
            raw_volume = min(raw_volume, max_by_margin)

        max_by_leverage = 0.0
        if self.config.max_leverage > 0:
            max_notional = equity * float(self.config.max_leverage)
            max_by_leverage = max_notional / max(price * self._symbol_contract_size, 1e-9)
            raw_volume = min(raw_volume, max_by_leverage)

        raw_volume = self._clamp(raw_volume, float(getattr(info, "volume_min", 0.01)), float(getattr(info, "volume_max", raw_volume)))
        step = float(getattr(info, "volume_step", 0.01))
        volume = self._round_volume(raw_volume, step)
        volume = min(volume, self.config.max_lots)
        if max_by_leverage > 0:
            volume = min(volume, max_by_leverage)
        return volume

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
            "comment": "doom-bias-demo" if not self.config.live else "doom-bias-live",
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
                mapping = {
                    0: self.mt5.ORDER_FILLING_FOK,
                    1: self.mt5.ORDER_FILLING_IOC,
                    2: self.mt5.ORDER_FILLING_RETURN,
                    3: self.mt5.ORDER_FILLING_BOC,
                }
                try:
                    raw_i = int(raw)
                    if raw_i in mapping:
                        candidates.append(mapping[raw_i])
                except Exception:
                    pass
        # Fallback order: IOC -> RETURN -> FOK. BOC is rarely valid for market orders.
        for mode in (self.mt5.ORDER_FILLING_IOC, self.mt5.ORDER_FILLING_RETURN, self.mt5.ORDER_FILLING_FOK):
            if mode not in candidates:
                candidates.append(mode)
        return candidates[0]

    def _is_trade_success(self, result: Any) -> bool:
        if result is None:
            return False
        code = getattr(result, "retcode", None)
        return code in {
            self.mt5.TRADE_RETCODE_DONE,
            self.mt5.TRADE_RETCODE_PLACED,
            self.mt5.TRADE_RETCODE_DONE_PARTIAL,
        }

    def _send_order_with_filling_fallback(self, request: Dict[str, Any]) -> Any:
        """Try several filling modes if the broker rejects the first one.

        Some FTMO / MT5 setups expose a symbol trade_fill_mode that does not
        work reliably for market execution through the Python bridge, so we
        retry with the common market-safe modes before failing.
        """
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
            self._log_file("order_send(type_filling=%s) -> %s", mode, self._result_to_dict(result))
            LOGGER.info("order_send(type_filling=%s) -> %s", mode, self._result_to_dict(result))
            if result is not None:
                code = getattr(result, "retcode", None)
                if code in {self.mt5.TRADE_RETCODE_DONE, self.mt5.TRADE_RETCODE_PLACED, self.mt5.TRADE_RETCODE_DONE_PARTIAL}:
                    return result
                # If the broker explicitly complains about filling mode, retry.
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
            "comment": "trail-adjust",
        }
        if self.config.live:
            result = self.mt5.order_send(request)
            LOGGER.info("SLTP modify result: %s", self._result_to_dict(result))
        else:
            LOGGER.info("DRY-RUN SLTP modify: %s", request)

    def close_all_positions(self) -> None:
        positions = self._positions()
        if not positions:
            LOGGER.info("No positions to close.")
            return
        ask, bid = self._tick_prices()
        for pos in positions:
            direction = "SELL" if pos.type == self.mt5.POSITION_TYPE_BUY else "BUY"
            price = bid if direction == "SELL" else ask
            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": self.mt5.ORDER_TYPE_SELL if pos.type == self.mt5.POSITION_TYPE_BUY else self.mt5.ORDER_TYPE_BUY,
                "position": pos.ticket,
                "price": price,
                "deviation": int(self.config.deviation),
                "magic": int(self.config.magic),
                "comment": "close-all",
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": self._select_filling_mode(),
            }
            if self.config.live:
                result = self._send_order_with_filling_fallback(request)
                LOGGER.info("Close result: %s", self._result_to_dict(result))
            else:
                LOGGER.info("DRY-RUN close request: %s", request)

    def _tick_prices(self) -> Tuple[float, float]:
        tick = self.mt5.symbol_info_tick(self.symbol)
        if tick is None:
            raise RuntimeError(f"symbol_info_tick failed: {self._last_error_safe()}")
        ask = float(getattr(tick, "ask", 0.0) or getattr(tick, "last", 0.0) or 0.0)
        bid = float(getattr(tick, "bid", 0.0) or getattr(tick, "last", 0.0) or 0.0)
        if ask <= 0 or bid <= 0:
            raise RuntimeError(f"Invalid bid/ask: ask={ask} bid={bid}")
        return ask, bid

    def _point(self) -> float:
        return float(self._symbol_point)

    def _digits(self) -> int:
        return int(self._symbol_digits)

    def _round_to_digits(self, value: float, digits: int) -> float:
        return round(value, digits)

    def _contract_value_per_point(self) -> float:
        return max(float(self._symbol_contract_size) * self._point(), 1e-9)

    def _score_to_strength(self, score: float) -> float:
        return self._clamp((abs(score) - self.config.trend_threshold) / max(1e-9, 1.5 - self.config.trend_threshold), 0.0, 1.0)

    def _interpolate(self, low: float, high: float, strength: float) -> float:
        return low + (high - low) * self._clamp(strength, 0.0, 1.0)

    def _round_volume(self, volume: float, step: float) -> float:
        if step <= 0:
            return round(volume, 2)
        return math.floor(volume / step) * step

    def _clamp(self, value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    def _result_to_dict(self, result: Any) -> Any:
        if result is None:
            return None
        if hasattr(result, "_asdict"):
            try:
                return result._asdict()
            except Exception:
                pass
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MT5 doomsday-style strategy (demo-safe defaults).")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M5")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18812)
    parser.add_argument("--live", action="store_true", help="Actually send orders.")
    parser.add_argument("--risk-pct", type=float, default=0.0075, help="Risk per trade as equity fraction.")
    parser.add_argument("--stop-atr", type=float, default=3.0, help="Stop loss distance in ATR.")
    parser.add_argument("--reward-multiple", type=float, default=2.4, help="Minimum reward multiple floor used to keep runners alive.")
    parser.add_argument("--tp-min-usd", type=float, default=24.0, help="Minimum TP target in USD, scaled by signal strength.")
    parser.add_argument("--tp-max-usd", type=float, default=72.0, help="Maximum TP target in USD, scaled by signal strength.")
    parser.add_argument("--long-bias", type=float, default=0.68, help="0.0..1.0, >0.5 favors buys.")
    parser.add_argument("--trend-threshold", type=float, default=0.60, help="Absolute score needed to trade.")
    parser.add_argument("--roll-trigger-pct", type=float, default=0.08, help="Roll / lock trigger as equity fraction.")
    parser.add_argument("--cooldown-minutes", type=int, default=25, help="Minimum minutes between closes and re-entries.")
    parser.add_argument("--max-leverage", type=float, default=7.5, help="Hard leverage cap used for sizing.")
    parser.add_argument("--max-drawdown-pct", type=float, default=0.05, help="Stop if peak-to-current DD exceeds this.")
    parser.add_argument("--max-daily-loss-pct", type=float, default=0.025, help="Stop if session loss exceeds this.")
    parser.add_argument("--max-lots", type=float, default=3.0)
    parser.add_argument("--magic", type=int, default=203493)
    parser.add_argument("--deviation", type=int, default=50)
    parser.add_argument("--loop-seconds", type=int, default=10)
    parser.add_argument("--lookback-bars", type=int, default=110)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--fast-sma", type=int, default=7)
    parser.add_argument("--slow-sma", type=int, default=30)
    parser.add_argument("--high-vol-only", action=argparse.BooleanOptionalAction, default=True, help="Only open new trades during high-volatility regimes.")
    parser.add_argument("--high-vol-atr-pct", type=float, default=0.0021, help="ATR/price threshold for high-volatility gating.")
    parser.add_argument("--high-vol-range-atr", type=float, default=4.75, help="20-bar range/ATR threshold for high-volatility gating.")
    parser.add_argument("--high-vol-breakout-lookback", type=int, default=16, help="Breakout lookback used by high-volatility momentum scoring.")
    parser.add_argument("--high-vol-min-momentum", type=float, default=0.85, help="Minimum absolute ATR-normalized momentum for high-volatility entries.")
    parser.add_argument("--high-vol-spike-atr", type=float, default=3.0, help="Reserved threshold for single-bar volatility spike filters.")
    parser.add_argument("--high-vol-min-breakout-atr", type=float, default=0.35, help="Minimum close-through breakout distance in ATR for entries.")
    parser.add_argument("--high-vol-min-close-location", type=float, default=0.68, help="Minimum close location for breakout candles; sells use the inverse.")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--terminal-path",
        default=r"C:\Program Files\MetaTrader 5\terminal64.exe",
        help="Windows terminal path inside the Wine prefix. Set empty to skip explicit initialize(path=...).",
    )
    parser.add_argument("--log-file", default="", help="Optional file path for strategy logs.")
    parser.add_argument("--state-path", default="", help="Optional file path for strategy state.")
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
        tp_min_usd=float(args.tp_min_usd),
        tp_max_usd=float(args.tp_max_usd),
        long_bias=float(args.long_bias),
        trend_threshold=float(args.trend_threshold),
        roll_trigger_pct=float(args.roll_trigger_pct),
        cooldown_minutes=int(args.cooldown_minutes),
        max_leverage=float(args.max_leverage),
        max_drawdown_pct=float(args.max_drawdown_pct),
        max_daily_loss_pct=float(args.max_daily_loss_pct),
        max_lots=float(args.max_lots),
        magic=int(args.magic),
        deviation=int(args.deviation),
        loop_seconds=int(args.loop_seconds),
        lookback_bars=int(args.lookback_bars),
        atr_period=int(args.atr_period),
        fast_sma=int(args.fast_sma),
        slow_sma=int(args.slow_sma),
        high_vol_only=bool(args.high_vol_only),
        high_vol_atr_pct=float(args.high_vol_atr_pct),
        high_vol_range_atr=float(args.high_vol_range_atr),
        high_vol_breakout_lookback=int(args.high_vol_breakout_lookback),
        high_vol_min_momentum=float(args.high_vol_min_momentum),
        high_vol_spike_atr=float(args.high_vol_spike_atr),
        high_vol_min_breakout_atr=float(args.high_vol_min_breakout_atr),
        high_vol_min_close_location=float(args.high_vol_min_close_location),
        log_level=args.log_level,
        terminal_path=terminal_path,
        log_file=args.log_file.strip() or None,
        state_path=args.state_path.strip() or None,
    )

    strategy = DoomsdayMT5Strategy(config)
    strategy.run()


if __name__ == "__main__":
    main()
