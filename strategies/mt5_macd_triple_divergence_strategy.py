#!/usr/bin/env python3.14
"""MT5 Python strategy — BanMuXia MACD Triple Divergence.

Based on the MACD triple divergence strategy described by 半木夏:

- MACD(12,26,9) histogram divergence detection
- 3 consecutive swing highs/lows with 2 consecutive MACD shortenings
- Opposite-color bar requirement between histogram segments
- Entry on next bar after confirmation
- Stop at 3rd peak high (short) / 3rd trough low (long)
- Additional exit: next bar histogram must continue shortening

Recommended timeframe: D1 (primary) or H4
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pymt5linux import MetaTrader5

LOGGER = logging.getLogger("macd_triple_div_strategy")
STATE_PATH_DEFAULT = "/home/chain4655/Documents/Sample/Python/macd_triple_div_state.json"
LOG_PATH_DEFAULT = "/home/chain4655/Documents/Sample/Python/macd_triple_div_strategy.log"


@dataclasses.dataclass(frozen=True)
class StrategyConfig:
    symbol: str
    timeframe: str
    host: str
    port: int
    live: bool
    start_equity: float
    daily_dd_limit: float
    total_dd_limit: float
    profit_target: float
    risk_pct: float
    max_lots: float
    max_leverage: float
    macd_fast: int
    macd_slow: int
    macd_signal: int
    swing_window: int
    atr_period: int
    stop_buffer_atr: float
    reward_multiple: float
    trail_trigger_atr: float
    trail_lock_atr: float
    max_spread_points: float
    max_trades_per_day: int
    max_consecutive_losses: int
    cooldown_bars_after_trade: int
    max_hold_bars: int
    loop_seconds: int
    lookback_bars: int
    state_path: str
    log_file: str
    terminal_path: Optional[str]
    deviation: int
    magic: int
    log_level: str


@dataclasses.dataclass
class MarketSnapshot:
    bar_time: dt.datetime
    close: float
    high: float
    low: float
    open: float
    atr: float
    macd_histogram: List[float]
    signal: str
    divergence_bar_idx: int
    divergence_peak_idx: int
    divergence_price: float
    divergence_stop: float
    spread_points: float = 0.0


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
    last_trade_bar_time: Optional[str] = None
    last_close_profit: float = 0.0
    paused_reason: str = ""
    paused: bool = False
    last_divergence_signal: str = "NONE"
    last_divergence_bar_idx: int = -1
    entry_bar_count: int = 0


class MACDTripleDivergenceStrategy:
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.mt5 = MetaTrader5(host=config.host, port=config.port)
        self.symbol = config.symbol
        self.timeframe = self._resolve_timeframe(config.timeframe)
        self.state_path = Path(config.state_path)
        self.log_file = Path(config.log_file)
        self.state = self._load_state()
        self._last_bar_time: Optional[dt.datetime] = None
        self._initialized = False
        self._rates_fail_streak = 0
        self._connect_fail_streak = 0
        self._last_reconnect_ts = 0.0
        self._symbol_point: float = 0.01
        self._symbol_digits: int = 2
        self._symbol_volume_min: float = 0.01
        self._symbol_volume_step: float = 0.01
        self._symbol_volume_max: float = 1.0
        self._symbol_contract_size: float = 1.0
        self._equity_fail_streak = 0

    def run(self) -> None:
        self._connect()
        self._prepare_symbol()

        while True:
            try:
                snapshot = self._build_snapshot()
                if not self._initialized:
                    self._ensure_day_context(snapshot.bar_time)
                    self._seed_equity()
                    self._log(
                        "Strategy started: %s %s live=%s MACD(%d,%d,%d)",
                        self.symbol,
                        self.config.timeframe,
                        self.config.live,
                        self.config.macd_fast,
                        self.config.macd_slow,
                        self.config.macd_signal,
                    )
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
                last_trade_bar_time=data.get("last_trade_bar_time"),
                last_close_profit=float(data.get("last_close_profit", 0.0)),
                paused_reason=str(data.get("paused_reason", "")),
                paused=bool(data.get("paused", False)),
                last_divergence_signal=str(data.get("last_divergence_signal", "NONE")),
                last_divergence_bar_idx=int(data.get("last_divergence_bar_idx", -1)),
                entry_bar_count=int(data.get("entry_bar_count", 0)),
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
            "last_trade_bar_time": self.state.last_trade_bar_time,
            "last_close_profit": self.state.last_close_profit,
            "paused_reason": self.state.paused_reason,
            "paused": self.state.paused,
            "last_divergence_signal": self.state.last_divergence_signal,
            "last_divergence_bar_idx": self.state.last_divergence_bar_idx,
            "entry_bar_count": self.state.entry_bar_count,
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
            raise RuntimeError(f"symbol_select failed for {self.symbol}: {self.mt5.last_error()}")

        info = self.mt5.symbol_info(self.symbol)
        if info is None:
            raise RuntimeError(f"symbol_info is None for {self.symbol}")

        self._symbol_digits = int(getattr(info, "digits", 2) or 2)
        self._symbol_point = float(getattr(info, "point", 0.01) or 0.01)
        self._symbol_volume_min = float(getattr(info, "volume_min", 0.01) or 0.01)
        self._symbol_volume_step = float(getattr(info, "volume_step", 0.01) or 0.01)
        self._symbol_volume_max = float(getattr(info, "volume_max", 1.0) or 1.0)
        self._symbol_contract_size = float(getattr(info, "trade_contract_size", 1.0) or 1.0)
        self._log(
            "Symbol ready: %s | digits=%s point=%s volume_min=%s volume_step=%s volume_max=%s",
            self.symbol,
            self._symbol_digits,
            self._symbol_point,
            self._symbol_volume_min,
            self._symbol_volume_step,
            self._symbol_volume_max,
        )

    def _shutdown(self) -> None:
        try:
            self.mt5.shutdown()
        except Exception:
            pass

    def _seed_equity(self) -> None:
        equity = self._get_equity()
        if self.state.initial_equity is None:
            self.state.initial_equity = equity
        if self.state.day_start_equity is None:
            self.state.day_start_equity = equity
        if self.state.max_equity_seen is None:
            self.state.max_equity_seen = equity
        self.state.last_equity = equity
        self._save_state()

    def _get_equity(self) -> float:
        for attempt in range(3):
            info = self.mt5.account_info()
            if info is not None:
                equity = getattr(info, "equity", None)
                if equity is None:
                    equity = getattr(info, "balance", None)
                if equity is not None:
                    self._equity_fail_streak = 0
                    return float(equity)
            self._equity_fail_streak += 1
            self._log(
                "account_info unavailable (attempt %s/3, streak=%s): %s",
                attempt + 1,
                self._equity_fail_streak,
                self.mt5.last_error(),
            )
            time.sleep(1)
            if self._equity_fail_streak < 3:
                continue
            try:
                self._handle_transient_ipc(RuntimeError("account_info unavailable"))
                self._equity_fail_streak = 0
                self._connect_fail_streak = 0
            except Exception as exc:
                self._connect_fail_streak += 1
                self._log("Reconnect attempt failed (streak=%s): %s", self._connect_fail_streak, exc)
        raise RuntimeError(f"account_info is None after reconnect attempts: {self.mt5.last_error()}")

    def _ensure_day_context(self, bar_time: dt.datetime) -> None:
        day_key = bar_time.date().isoformat()
        if self.state.current_day != day_key:
            equity = self._get_equity()
            if self.state.current_day is not None and self.state.day_start_equity is not None:
                day_profit = equity - float(self.state.day_start_equity)
                self._log("Day rollover | previous_day=%s profit=%.2f", self.state.current_day, day_profit)
            self.state.current_day = day_key
            self.state.day_start_equity = equity
            self.state.trades_today = 0
            self.state.consecutive_losses = 0
            self.state.last_trade_bar_time = None
            self.state.paused = False
            self.state.paused_reason = ""
            self._save_state()

    def _risk_guard(self, snapshot: MarketSnapshot) -> None:
        equity = self._get_equity()
        self.state.last_equity = equity
        self.state.max_equity_seen = max(float(self.state.max_equity_seen or equity), equity)

        initial_equity = float(self.state.initial_equity or self.config.start_equity)
        day_start_equity = float(self.state.day_start_equity or equity)
        daily_dd = max(0.0, (day_start_equity - equity) / max(day_start_equity, 1e-9))
        total_dd = max(0.0, (initial_equity - equity) / max(initial_equity, 1e-9))
        profit_progress = (equity / max(initial_equity, 1e-9)) - 1.0

        self._log(
            "Risk | equity=%.2f day_dd=%.2f%% total_dd=%.2f%% profit=%.2f%% paused=%s",
            equity,
            daily_dd * 100.0,
            total_dd * 100.0,
            profit_progress * 100.0,
            self.state.paused,
        )
        self._save_state()

        if daily_dd >= self.config.daily_dd_limit:
            self.state.paused = True
            self.state.paused_reason = (
                f"Daily drawdown {daily_dd * 100.0:.2f}% >= {self.config.daily_dd_limit * 100.0:.2f}%"
            )
            self._log(self.state.paused_reason)
            self.close_all_positions()
            self._save_state()
            return

        if total_dd >= self.config.total_dd_limit:
            self.state.paused = True
            self.state.paused_reason = (
                f"Total drawdown {total_dd * 100.0:.2f}% >= {self.config.total_dd_limit * 100.0:.2f}%"
            )
            self._log(self.state.paused_reason)
            self.close_all_positions()
            self._save_state()
            return

        if profit_progress >= self.config.profit_target:
            self.state.paused = True
            self.state.paused_reason = f"Profit target reached: {profit_progress * 100.0:.2f}%"
            self._log(self.state.paused_reason)
            self.close_all_positions()
            self._save_state()
            return

    def _build_snapshot(self) -> MarketSnapshot:
        warmup = max(self.config.macd_slow, self.config.atr_period) * 3 + self.config.swing_window * 3
        need = max(self.config.lookback_bars, warmup)
        bars = self._fetch_bars(min_count=need)
        if len(bars) < need:
            raise RuntimeError(
                f"Not enough bars: got={len(bars)} need={need} lookback={self.config.lookback_bars}"
            )

        bars_closed = bars[:-1]
        closes = [bar["close"] for bar in bars_closed]
        highs = [bar["high"] for bar in bars_closed]
        lows = [bar["low"] for bar in bars_closed]
        last_closed = bars_closed[-1]

        atr = self._atr(bars_closed, self.config.atr_period)
        macd_line, signal_line, histogram = self._calc_macd(closes)
        signal, div_bar_idx, div_peak_idx, div_price, div_stop = self._detect_divergence(
            closes, highs, lows, histogram
        )

        bar_time = self._bar_time(last_closed)
        spread = self._spread_points()
        return MarketSnapshot(
            bar_time=bar_time,
            close=float(last_closed["close"]),
            high=float(last_closed["high"]),
            low=float(last_closed["low"]),
            open=float(last_closed["open"]),
            atr=atr,
            macd_histogram=histogram,
            signal=signal,
            divergence_bar_idx=div_bar_idx,
            divergence_peak_idx=div_peak_idx,
            divergence_price=div_price,
            divergence_stop=div_stop,
            spread_points=spread,
        )

    def _fetch_bars(self, min_count: Optional[int] = None) -> List[Dict[str, float]]:
        fetch_count = max(int(self.config.lookback_bars), int(min_count or 0))
        while True:
            for attempt in range(3):
                rates = self.mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, fetch_count)
                if rates is not None:
                    self._rates_fail_streak = 0
                    return [self._normalize_bar(row) for row in list(rates)]

                self._rates_fail_streak += 1
                self._log(
                    "copy_rates_from_pos unavailable (attempt %s/3, streak=%s): %s",
                    attempt + 1,
                    self._rates_fail_streak,
                    self.mt5.last_error(),
                )
                time.sleep(1)
                if self._rates_fail_streak < 3:
                    continue
                try:
                    self._handle_transient_ipc(RuntimeError("copy_rates_from_pos unavailable"))
                    self._rates_fail_streak = 0
                    self._connect_fail_streak = 0
                except Exception as exc:
                    self._connect_fail_streak += 1
                    self._log("Reconnect after rates failure failed (streak=%s): %s", self._connect_fail_streak, exc)
                break

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

    def _calc_macd(self, closes: Sequence[float]) -> Tuple[List[float], List[float], List[float]]:
        ema_fast = self._ema(closes, self.config.macd_fast)
        ema_slow = self._ema(closes, self.config.macd_slow)
        macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
        signal_line = self._ema(macd_line, self.config.macd_signal)
        histogram = [m - s for m, s in zip(macd_line, signal_line)]
        return macd_line, signal_line, histogram

    def _ema(self, values: Sequence[float], period: int) -> List[float]:
        result: List[float] = []
        if len(values) < period:
            return result
        multiplier = 2.0 / (period + 1)
        ema = statistics.fmean(values[:period])
        result.append(ema)
        for v in values[period:]:
            ema = (v - ema) * multiplier + ema
            result.append(ema)
        return result

    def _find_swing_highs(self, highs: Sequence[float]) -> List[int]:
        window = self.config.swing_window
        indices: List[int] = []
        for i in range(window, len(highs) - window):
            if all(highs[i] > highs[i - j] for j in range(1, window + 1)) and \
               all(highs[i] >= highs[i + j] for j in range(1, window + 1)):
                indices.append(i)
        return indices

    def _find_swing_lows(self, lows: Sequence[float]) -> List[int]:
        window = self.config.swing_window
        indices: List[int] = []
        for i in range(window, len(lows) - window):
            if all(lows[i] < lows[i - j] for j in range(1, window + 1)) and \
               all(lows[i] <= lows[i + j] for j in range(1, window + 1)):
                indices.append(i)
        return indices

    def _opposite_color_between(self, hist: Sequence[float], idx1: int, idx2: int) -> bool:
        if idx1 >= idx2:
            return False
        segment = hist[idx1 + 1: idx2]
        if not segment:
            return False
        if hist[idx1] >= 0:
            return any(h < 0 for h in segment)
        else:
            return any(h > 0 for h in segment)

    def _detect_divergence(
        self,
        closes: Sequence[float],
        highs: Sequence[float],
        lows: Sequence[float],
        histogram: Sequence[float],
    ) -> Tuple[str, int, int, float, float]:
        swing_highs = self._find_swing_highs(highs)
        swing_lows = self._find_swing_lows(lows)

        div_bar_idx = -1
        div_peak_idx = -1
        div_price = 0.0
        div_stop = 0.0

        last_closed_idx = len(highs) - 1

        for i in range(len(swing_highs) - 2):
            h1, h2, h3 = swing_highs[i], swing_highs[i + 1], swing_highs[i + 2]
            if h3 != last_closed_idx:
                continue
            if not (highs[h3] > highs[h2] > highs[h1]):
                continue
            if not (histogram[h3] < histogram[h2] < histogram[h1]):
                continue
            if not (histogram[h1] > 0 and histogram[h2] > 0 and histogram[h3] > 0):
                continue
            if not self._opposite_color_between(histogram, h1, h2):
                continue
            if not self._opposite_color_between(histogram, h2, h3):
                continue
            div_bar_idx = h3
            div_peak_idx = h3
            div_price = highs[h3]
            div_stop = highs[h3]
            self._log(
                "SELL divergence detected: peaks at idx=%d,%d,%d prices=%.2f,%.2f,%.2f hist=%.4f,%.4f,%.4f",
                h1, h2, h3,
                highs[h1], highs[h2], highs[h3],
                histogram[h1], histogram[h2], histogram[h3],
            )
            return "SELL", div_bar_idx, div_peak_idx, div_price, div_stop

        for i in range(len(swing_lows) - 2):
            l1, l2, l3 = swing_lows[i], swing_lows[i + 1], swing_lows[i + 2]
            if l3 != last_closed_idx:
                continue
            if not (lows[l3] < lows[l2] < lows[l1]):
                continue
            if not (histogram[l3] > histogram[l2] > histogram[l1]):
                continue
            if not (histogram[l1] < 0 and histogram[l2] < 0 and histogram[l3] < 0):
                continue
            if not self._opposite_color_between(histogram, l1, l2):
                continue
            if not self._opposite_color_between(histogram, l2, l3):
                continue
            div_bar_idx = l3
            div_peak_idx = l3
            div_price = lows[l3]
            div_stop = lows[l3]
            self._log(
                "BUY divergence detected: troughs at idx=%d,%d,%d prices=%.2f,%.2f,%.2f hist=%.4f,%.4f,%.4f",
                l1, l2, l3,
                lows[l1], lows[l2], lows[l3],
                histogram[l1], histogram[l2], histogram[l3],
            )
            return "BUY", div_bar_idx, div_peak_idx, div_price, div_stop

        return "NONE", -1, -1, 0.0, 0.0

    def _handle_bar(self, snapshot: MarketSnapshot) -> None:
        positions = self._positions()
        foreign_positions = self._foreign_positions()
        hist_current = snapshot.macd_histogram[-1] if snapshot.macd_histogram else 0.0
        hist_prev = snapshot.macd_histogram[-2] if len(snapshot.macd_histogram) >= 2 else 0.0

        self._log(
            "Bar %s | close=%.2f atr=%.2f signal=%s hist=%.4f positions=%d foreign=%d trades_today=%d losses=%d",
            snapshot.bar_time,
            snapshot.close,
            snapshot.atr,
            snapshot.signal,
            hist_current,
            len(positions),
            len(foreign_positions),
            self.state.trades_today,
            self.state.consecutive_losses,
        )

        if self.state.paused:
            self._log("Trading paused: %s", self.state.paused_reason)
            return

        if snapshot.spread_points > self.config.max_spread_points:
            self._log("Spread filter blocked entry: %.1f > %.1f", snapshot.spread_points, self.config.max_spread_points)
            if positions:
                self._maybe_manage_position(snapshot, positions[0])
            return

        if self.state.trades_today >= self.config.max_trades_per_day:
            self._log("Daily trade cap reached: %d", self.state.trades_today)
            if positions:
                self._maybe_manage_position(snapshot, positions[0])
            return

        if self.state.consecutive_losses >= self.config.max_consecutive_losses:
            self._log("Consecutive loss cap reached: %d", self.state.consecutive_losses)
            if positions:
                self._maybe_manage_position(snapshot, positions[0])
            return

        if not positions:
            if foreign_positions:
                self._log("Foreign position present (%d); skip new entry.", len(foreign_positions))
                return
            if snapshot.signal != "NONE" and self._cooldown_ok(snapshot.bar_time):
                self._enter(snapshot)
            return

        pos = positions[0]
        self.state.entry_bar_count += 1

        if self.state.entry_bar_count >= self.config.max_hold_bars:
            self._log("Max hold bars exceeded (%d), closing position.", self.state.entry_bar_count)
            self.close_all_positions()
            return

        if self._should_exit_on_histogram(snapshot, pos):
            self._log("Histogram failed to continue shortening, exiting position.")
            self.close_all_positions()
            return

        if self._should_reverse(snapshot, pos):
            self._log("Opposite divergence detected, closing current position.")
            self.close_all_positions()
            time.sleep(1)
            if not self.state.paused and self._cooldown_ok(snapshot.bar_time):
                self._enter(snapshot)
            return

        self._maybe_manage_position(snapshot, pos)

    def _spread_points(self) -> float:
        tick = self.mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return 999999.0
        ask = float(getattr(tick, "ask", 0.0) or 0.0)
        bid = float(getattr(tick, "bid", 0.0) or 0.0)
        if ask <= 0 or bid <= 0:
            return 999999.0
        return abs(ask - bid) / max(self._point(), 1e-9)

    def _all_positions(self) -> List[PositionState]:
        raw = self.mt5.positions_get(symbol=self.symbol)
        if not raw:
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
        return out

    def _foreign_positions(self) -> List[PositionState]:
        return [pos for pos in self._all_positions() if int(getattr(pos, "magic", 0)) != int(self.config.magic)]

    def _positions(self) -> List[PositionState]:
        return [pos for pos in self._all_positions() if int(getattr(pos, "magic", 0)) == int(self.config.magic)]

    def _should_exit_on_histogram(self, snapshot: MarketSnapshot, pos: PositionState) -> bool:
        if len(snapshot.macd_histogram) < 3:
            return False
        h_prev = snapshot.macd_histogram[-2]
        h_curr = snapshot.macd_histogram[-1]
        if pos.type == self.mt5.POSITION_TYPE_BUY:
            return h_curr <= h_prev
        else:
            return h_curr >= h_prev

    def _should_reverse(self, snapshot: MarketSnapshot, pos: PositionState) -> bool:
        if pos.type == self.mt5.POSITION_TYPE_BUY and snapshot.signal == "SELL":
            return True
        if pos.type == self.mt5.POSITION_TYPE_SELL and snapshot.signal == "BUY":
            return True
        return False

    def _maybe_manage_position(self, snapshot: MarketSnapshot, pos: PositionState) -> None:
        if snapshot.atr <= 0:
            return
        ask, bid = self._tick_prices()
        current_price = bid if pos.type == self.mt5.POSITION_TYPE_BUY else ask
        profit_move = (current_price - pos.price_open) if pos.type == self.mt5.POSITION_TYPE_BUY else (pos.price_open - current_price)

        if profit_move < snapshot.atr * self.config.trail_trigger_atr:
            return

        if pos.type == self.mt5.POSITION_TYPE_BUY:
            new_sl = max(pos.sl, pos.price_open + snapshot.atr * self.config.trail_lock_atr)
            if pos.sl <= 0 or new_sl > pos.sl:
                self._modify_position(pos, sl=self._round_to_digits(new_sl, self._digits()), tp=pos.tp)
        else:
            new_sl = min(pos.sl if pos.sl > 0 else current_price + snapshot.atr * 100.0,
                         pos.price_open - snapshot.atr * self.config.trail_lock_atr)
            if pos.sl <= 0 or new_sl < pos.sl:
                self._modify_position(pos, sl=self._round_to_digits(new_sl, self._digits()), tp=pos.tp)

    def _enter(self, snapshot: MarketSnapshot) -> None:
        direction = snapshot.signal
        if direction not in {"BUY", "SELL"}:
            return

        ask, bid = self._tick_prices()
        price = ask if direction == "BUY" else bid
        sl, tp = self._build_sl_tp(direction, price, snapshot)
        volume = self._size_position(direction, price, sl)
        if volume <= 0:
            raise RuntimeError("Calculated volume is zero")

        request = self._order_request(direction, volume, price, sl, tp)
        self._log(
            "ENTRY %s volume=%.2f price=%.2f sl=%.2f tp=%.2f equity=%.2f signal=%s div_bar=%d",
            direction,
            volume,
            price,
            sl,
            tp,
            self._get_equity(),
            snapshot.signal,
            snapshot.divergence_bar_idx,
        )

        if self.config.live:
            result = self._send_order_with_filling_fallback(request)
            self._log("order_send result: %s", self._result_to_dict(result))
            if result is None:
                raise RuntimeError(f"order_send failed: {self.mt5.last_error()}")
        else:
            self._log("DRY-RUN request: %s", request)

        self.state.trades_today += 1
        self.state.last_trade_bar_time = snapshot.bar_time.isoformat()
        self.state.last_divergence_signal = snapshot.signal
        self.state.last_divergence_bar_idx = snapshot.divergence_bar_idx
        self.state.entry_bar_count = 0
        self._save_state()

    def _build_sl_tp(self, direction: str, price: float, snapshot: MarketSnapshot) -> Tuple[float, float]:
        stop_distance = abs(price - snapshot.divergence_stop)
        sl_buffer = max(snapshot.atr * self.config.stop_buffer_atr, self._point() * 15)
        risk_distance = max(stop_distance, sl_buffer)
        tp_distance = risk_distance * self.config.reward_multiple
        digits = self._digits()

        if direction == "BUY":
            sl = min(snapshot.divergence_stop - sl_buffer, price - risk_distance * 0.5)
            tp = price + tp_distance
        else:
            sl = max(snapshot.divergence_stop + sl_buffer, price + risk_distance * 0.5)
            tp = price - tp_distance

        sl = self._round_to_digits(sl, digits)
        tp = self._round_to_digits(tp, digits)
        point = self._point()
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
        max_vol = float(getattr(info, "volume_max", raw_volume))
        raw_volume = self._clamp(raw_volume, min_vol, max_vol)
        step = float(getattr(info, "volume_step", 0.01))
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
            "comment": "macd-triple-div-live" if self.config.live else "macd-triple-div-dryrun",
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
            self._log("order_send(type_filling=%s) -> %s", mode, self._result_to_dict(result))
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
            "comment": "div-trail-adjust",
        }
        if self.config.live:
            result = self.mt5.order_send(request)
            self._log("SLTP modify result: %s", self._result_to_dict(result))
            if result is None:
                self._log("SLTP modify failed: %s", self.mt5.last_error())
        else:
            self._log("DRY-RUN SLTP modify: %s", request)

    def close_all_positions(self) -> None:
        positions = self._positions()
        if not positions:
            self._log("No positions to close.")
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
                "comment": "div-close-all",
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": self._select_filling_mode(),
            }
            if self.config.live:
                result = self._send_order_with_filling_fallback(request)
                self._log("Close result: %s", self._result_to_dict(result))
            else:
                self._log("DRY-RUN close request: %s", request)

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
        return float(self._symbol_point)

    def _digits(self) -> int:
        return int(self._symbol_digits)

    def _round_to_digits(self, value: float, digits: int) -> float:
        return round(float(value), digits)

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
            return result._asdict()
        return str(result)

    def _cooldown_ok(self, bar_time: dt.datetime) -> bool:
        if self.state.last_trade_bar_time is None:
            return True
        try:
            last_dt = dt.datetime.fromisoformat(str(self.state.last_trade_bar_time))
        except Exception:
            return True
        bar_seconds = self._timeframe_seconds()
        delta_bars = (bar_time - last_dt).total_seconds() / max(bar_seconds, 1)
        return delta_bars >= float(self.config.cooldown_bars_after_trade)

    def _timeframe_seconds(self) -> float:
        key = str(self.config.timeframe).strip().upper()
        if key.startswith("M"):
            try:
                return float(max(1, int(key[1:])) * 60)
            except ValueError:
                return 300.0
        if key.startswith("H"):
            try:
                return float(max(1, int(key[1:])) * 3600)
            except ValueError:
                return 3600.0
        if key.startswith("D"):
            return 86400.0
        return float(max(self.config.loop_seconds, 1))

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

    def _is_transient_ipc_issue(self, exc: Exception) -> bool:
        text = str(exc).lower()
        keywords = ("broken pipe", "connection reset", "transport", "ipc", "timeout", "dead object")
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
        self._connect()
        self._prepare_symbol()
        self._connect_fail_streak = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MACD Triple Divergence MT5 Python strategy (半木夏).")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="D1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18812)
    parser.add_argument("--live", action="store_true", help="Actually send orders.")
    parser.add_argument("--start-equity", type=float, default=10000.0)
    parser.add_argument("--daily-dd-limit", type=float, default=0.03)
    parser.add_argument("--total-dd-limit", type=float, default=0.10)
    parser.add_argument("--profit-target", type=float, default=0.05)
    parser.add_argument("--risk-pct", type=float, default=0.005)
    parser.add_argument("--max-lots", type=float, default=2.0)
    parser.add_argument("--max-leverage", type=float, default=5.0)
    parser.add_argument("--macd-fast", type=int, default=12)
    parser.add_argument("--macd-slow", type=int, default=26)
    parser.add_argument("--macd-signal", type=int, default=9)
    parser.add_argument("--swing-window", type=int, default=3, help="Bars each side to confirm swing high/low.")
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--stop-buffer-atr", type=float, default=0.5, help="SL buffer in ATR beyond divergence point.")
    parser.add_argument("--reward-multiple", type=float, default=2.5)
    parser.add_argument("--trail-trigger-atr", type=float, default=1.5, help="ATR profit threshold to start trailing.")
    parser.add_argument("--trail-lock-atr", type=float, default=0.3, help="ATR distance to lock trailing profit.")
    parser.add_argument("--max-spread-points", type=float, default=120.0)
    parser.add_argument("--max-trades-per-day", type=int, default=1)
    parser.add_argument("--max-consecutive-losses", type=int, default=2)
    parser.add_argument("--cooldown-bars-after-trade", type=int, default=5)
    parser.add_argument("--max-hold-bars", type=int, default=50, help="Max bars to hold a position (per strategy validity).")
    parser.add_argument("--loop-seconds", type=int, default=60)
    parser.add_argument("--lookback-bars", type=int, default=500)
    parser.add_argument("--state-path", default=STATE_PATH_DEFAULT)
    parser.add_argument("--log-file", default=LOG_PATH_DEFAULT)
    parser.add_argument(
        "--terminal-path",
        default=r"C:\Program Files\MetaTrader 5\terminal64.exe",
        help="Windows terminal path inside the Wine prefix.",
    )
    parser.add_argument("--deviation", type=int, default=30)
    parser.add_argument("--magic", type=int, default=206482)
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
        start_equity=float(args.start_equity),
        daily_dd_limit=float(args.daily_dd_limit),
        total_dd_limit=float(args.total_dd_limit),
        profit_target=float(args.profit_target),
        risk_pct=float(args.risk_pct),
        max_lots=float(args.max_lots),
        max_leverage=float(args.max_leverage),
        macd_fast=int(args.macd_fast),
        macd_slow=int(args.macd_slow),
        macd_signal=int(args.macd_signal),
        swing_window=int(args.swing_window),
        atr_period=int(args.atr_period),
        stop_buffer_atr=float(args.stop_buffer_atr),
        reward_multiple=float(args.reward_multiple),
        trail_trigger_atr=float(args.trail_trigger_atr),
        trail_lock_atr=float(args.trail_lock_atr),
        max_spread_points=float(args.max_spread_points),
        max_trades_per_day=int(args.max_trades_per_day),
        max_consecutive_losses=int(args.max_consecutive_losses),
        cooldown_bars_after_trade=int(args.cooldown_bars_after_trade),
        max_hold_bars=int(args.max_hold_bars),
        loop_seconds=int(args.loop_seconds),
        lookback_bars=int(args.lookback_bars),
        state_path=args.state_path,
        log_file=args.log_file,
        terminal_path=terminal_path,
        deviation=int(args.deviation),
        magic=int(args.magic),
        log_level=args.log_level,
    )

    strategy = MACDTripleDivergenceStrategy(config)
    strategy.run()


if __name__ == "__main__":
    main()
