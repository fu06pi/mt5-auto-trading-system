#!/usr/bin/env python3.14
"""XAUUSD Asia -> London breakout MT5 Python strategy.

EP1 idea:
- Build the Asian session range first.
- Only trade during the London breakout window.
- Enter on confirmed breakout above/below the Asian range.
- Keep prop-firm style hard risk controls.

Default timing uses UTC windows.
If your broker server time differs, use --time-offset-hours to align bars.
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

LOGGER = logging.getLogger("xauusd_ep1_strategy")
STATE_PATH_DEFAULT = "/home/chain4655/Documents/Sample/Python/xauusd_ep1_state.json"
LOG_PATH_DEFAULT = "/home/chain4655/Documents/Sample/Python/xauusd_ep1_strategy.log"


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
    asia_start_utc: int
    asia_end_utc: int
    london_start_utc: int
    london_end_utc: int
    time_offset_hours: int
    atr_period: int
    breakout_buffer_atr: float
    stop_buffer_atr: float
    reward_multiple: float
    max_spread_points: float
    max_trades_per_day: int
    max_consecutive_losses: int
    cooldown_bars_after_trade: int
    max_hold_minutes: int
    loop_seconds: int
    lookback_bars: int
    htf_lookback_bars: int
    htf_fast_sma: int
    htf_slow_sma: int
    min_asia_range_atr: float
    max_asia_range_atr: float
    state_path: str
    log_file: str
    terminal_path: Optional[str]
    deviation: int
    magic: int
    log_level: str


@dataclasses.dataclass
class SessionRange:
    session_date: str
    high: float
    low: float
    open_price: float
    close_price: float
    bars: int


@dataclasses.dataclass
class MarketSnapshot:
    bar_time: dt.datetime
    close: float
    open: float
    high: float
    low: float
    atr: float
    htf_fast_sma: float
    htf_slow_sma: float
    htf_signal: str
    spread_points: float
    asia_high: float
    asia_low: float
    asia_width: float
    breakout_signal: str
    score: float


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
    last_processed_deal_time: Optional[str] = None
    last_close_profit: float = 0.0
    paused_reason: str = ""
    paused: bool = False


class XAUUSDAsiaLondonBreakoutStrategy:
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
                        "Strategy started: %s %s live=%s",
                        self.symbol,
                        self.config.timeframe,
                        self.config.live,
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
                last_processed_deal_time=data.get("last_processed_deal_time"),
                last_close_profit=float(data.get("last_close_profit", 0.0)),
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
            "last_trade_bar_time": self.state.last_trade_bar_time,
            "last_processed_deal_time": self.state.last_processed_deal_time,
            "last_close_profit": self.state.last_close_profit,
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
            raise RuntimeError(f"symbol_select failed for {self.symbol}: {self.mt5.last_error()}")
        self._log("Symbol prepared: %s", self.symbol)

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
            raise SystemExit(1)

        if total_dd >= self.config.total_dd_limit:
            self.state.paused = True
            self.state.paused_reason = (
                f"Total drawdown {total_dd * 100.0:.2f}% >= {self.config.total_dd_limit * 100.0:.2f}%"
            )
            self._log(self.state.paused_reason)
            self.close_all_positions()
            self._save_state()
            raise SystemExit(1)

        if profit_progress >= self.config.profit_target:
            self.state.paused = True
            self.state.paused_reason = f"Profit target reached: {profit_progress * 100.0:.2f}%"
            self._log(self.state.paused_reason)
            self.close_all_positions()
            self._save_state()
            raise SystemExit(0)

    def _build_snapshot(self) -> MarketSnapshot:
        need = max(self.config.atr_period, self.config.htf_slow_sma, self.config.htf_fast_sma) + 10
        bars = self._fetch_bars(min_count=need)
        if len(bars) < need:
            raise RuntimeError(
                f"Not enough bars for signal calculation: got={len(bars)} need={need} lookback={self.config.lookback_bars}"
            )

        last_closed = bars[-2]
        closes = [bar["close"] for bar in bars[:-1]]
        highs = [bar["high"] for bar in bars[:-1]]
        lows = [bar["low"] for bar in bars[:-1]]
        opens = [bar["open"] for bar in bars[:-1]]

        bar_time = self._adjust_time(self._bar_time(last_closed))
        atr = self._atr(bars[:-1], self.config.atr_period)
        htf_fast_sma, htf_slow_sma, htf_signal = self._build_htf_filter()
        spread_points = self._spread_points()
        asia = self._build_asia_range(bars, bar_time)

        breakout_signal, score = self._score_signal(
            closes=closes,
            opens=opens,
            atr=atr,
            htf_signal=htf_signal,
            asia=asia,
            bar_time=bar_time,
        )

        asia_high = asia.high if asia is not None else 0.0
        asia_low = asia.low if asia is not None else 0.0
        asia_width = (asia_high - asia_low) if asia is not None else 0.0

        return MarketSnapshot(
            bar_time=bar_time,
            close=float(last_closed["close"]),
            open=float(last_closed["open"]),
            high=float(last_closed["high"]),
            low=float(last_closed["low"]),
            atr=atr,
            htf_fast_sma=htf_fast_sma,
            htf_slow_sma=htf_slow_sma,
            htf_signal=htf_signal,
            spread_points=spread_points,
            asia_high=asia_high,
            asia_low=asia_low,
            asia_width=asia_width,
            breakout_signal=breakout_signal,
            score=score,
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

    def _adjust_time(self, bar_time: dt.datetime) -> dt.datetime:
        return bar_time + dt.timedelta(hours=int(self.config.time_offset_hours))

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

    def _build_htf_filter(self) -> Tuple[float, float, str]:
        htf_need = max(self.config.htf_lookback_bars, self.config.htf_slow_sma, self.config.htf_fast_sma) + 5
        bars = self.mt5.copy_rates_from_pos(self.symbol, self.mt5.TIMEFRAME_H1, 0, htf_need)
        if bars is None:
            raise RuntimeError(f"HTF copy_rates_from_pos failed: {self.mt5.last_error()}")
        htf_bars = [self._normalize_bar(row) for row in list(bars)]
        if len(htf_bars) < max(self.config.htf_slow_sma, self.config.htf_fast_sma) + 5:
            raise RuntimeError(
                f"Not enough HTF bars: got={len(htf_bars)} need={max(self.config.htf_slow_sma, self.config.htf_fast_sma) + 5} htf_need={htf_need}"
            )
        closes = [bar["close"] for bar in htf_bars[:-1]]
        fast = statistics.fmean(closes[-self.config.htf_fast_sma:])
        slow = statistics.fmean(closes[-self.config.htf_slow_sma:])
        if fast > slow:
            return fast, slow, "BULL"
        if fast < slow:
            return fast, slow, "BEAR"
        return fast, slow, "NEUTRAL"

    def _build_asia_range(self, bars: Sequence[Dict[str, float]], bar_time: dt.datetime) -> Optional[SessionRange]:
        day_key = bar_time.date().isoformat()
        selected: List[Dict[str, float]] = []
        for bar in bars:
            t = self._adjust_time(self._bar_time(bar))
            if t.date().isoformat() != day_key:
                continue
            if self._session_allowed(t, self.config.asia_start_utc, self.config.asia_end_utc):
                selected.append(bar)
        if len(selected) < 3:
            return None

        highs = [bar["high"] for bar in selected]
        lows = [bar["low"] for bar in selected]
        open_price = selected[0]["open"]
        close_price = selected[-1]["close"]
        return SessionRange(
            session_date=day_key,
            high=max(highs),
            low=min(lows),
            open_price=open_price,
            close_price=close_price,
            bars=len(selected),
        )

    def _score_signal(
        self,
        closes: Sequence[float],
        opens: Sequence[float],
        atr: float,
        htf_signal: str,
        asia: Optional[SessionRange],
        bar_time: dt.datetime,
    ) -> Tuple[str, float]:
        if asia is None or atr <= 0:
            return "NONE", 0.0
        if not self._session_allowed(bar_time, self.config.london_start_utc, self.config.london_end_utc):
            return "NONE", 0.0

        last_close = closes[-1]
        last_open = opens[-1]
        prev_close = closes[-2]
        breakout_buffer = max(atr * self.config.breakout_buffer_atr, self._point() * 10)
        range_width = asia.high - asia.low
        if range_width <= 0:
            return "NONE", 0.0

        bullish_break = last_close > asia.high + breakout_buffer and last_close > last_open and last_close >= prev_close
        bearish_break = last_close < asia.low - breakout_buffer and last_close < last_open and last_close <= prev_close

        if bullish_break and range_width < atr * self.config.min_asia_range_atr:
            return "NONE", 0.0
        if bearish_break and range_width < atr * self.config.min_asia_range_atr:
            return "NONE", 0.0
        if range_width > atr * self.config.max_asia_range_atr:
            return "NONE", 0.0

        htf_bias = 0.0
        if htf_signal == "BULL":
            htf_bias = 0.35
        elif htf_signal == "BEAR":
            htf_bias = -0.35

        momentum = self._clamp((last_close - last_open) / atr, -1.5, 1.5) * 0.25
        range_break = 0.0
        if bullish_break:
            range_break = 0.75
        elif bearish_break:
            range_break = -0.75

        score = self._clamp(range_break + htf_bias + momentum, -1.5, 1.5)

        if bullish_break and score >= 0.55:
            return "BUY", score
        if bearish_break and score <= -0.55:
            return "SELL", score
        return "NONE", score

    def _handle_bar(self, snapshot: MarketSnapshot) -> None:
        self._sync_closed_trades()
        positions = self._positions()
        foreign_positions = self._foreign_positions()
        self._log(
            "Bar %s | close=%.2f atr=%.2f htf=%s spread=%.1f asiaH=%.2f asiaL=%.2f width=%.2f signal=%s score=%.2f positions=%d foreign=%d trades_today=%d losses=%d",
            snapshot.bar_time,
            snapshot.close,
            snapshot.atr,
            snapshot.htf_signal,
            snapshot.spread_points,
            snapshot.asia_high,
            snapshot.asia_low,
            snapshot.asia_width,
            snapshot.breakout_signal,
            snapshot.score,
            len(positions),
            len(foreign_positions),
            self.state.trades_today,
            self.state.consecutive_losses,
        )

        if self.state.paused:
            self._log("Trading paused: %s", self.state.paused_reason)
            return

        if not self._trade_window_allowed(snapshot.bar_time):
            if positions:
                self._maybe_manage_position(snapshot, positions[0])
                if snapshot.bar_time.hour >= self.config.london_end_utc:
                    self._log("London window ended, closing position.")
                    self.close_all_positions()
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
            if snapshot.breakout_signal != "NONE" and self._cooldown_ok(snapshot.bar_time):
                self._enter(snapshot)
            return

        pos = positions[0]
        if self._max_hold_exceeded(pos, snapshot.bar_time):
            self._log("Max holding time exceeded, closing position.")
            self.close_all_positions()
            return

        if self._should_reverse(snapshot, pos):
            self._log("Opposite breakout detected, closing current position first.")
            self.close_all_positions()
            time.sleep(1)
            if not self.state.paused and self._cooldown_ok(snapshot.bar_time):
                self._enter(snapshot)
            return

        self._maybe_manage_position(snapshot, pos)

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

    def _symbol_positions(self) -> List[PositionState]:
        return self._all_positions()

    def _foreign_positions(self) -> List[PositionState]:
        return []

    def _positions(self) -> List[PositionState]:
        return [pos for pos in self._all_positions() if int(getattr(pos, "magic", 0)) == int(self.config.magic)]

    def _should_reverse(self, snapshot: MarketSnapshot, pos: PositionState) -> bool:
        if pos.type == self.mt5.POSITION_TYPE_BUY and snapshot.breakout_signal == "SELL":
            return True
        if pos.type == self.mt5.POSITION_TYPE_SELL and snapshot.breakout_signal == "BUY":
            return True
        return False

    def _maybe_manage_position(self, snapshot: MarketSnapshot, pos: PositionState) -> None:
        if snapshot.atr <= 0:
            return
        ask, bid = self._tick_prices()
        current_price = bid if pos.type == self.mt5.POSITION_TYPE_BUY else ask
        profit_move = (current_price - pos.price_open) if pos.type == self.mt5.POSITION_TYPE_BUY else (pos.price_open - current_price)

        if profit_move >= snapshot.atr:
            if pos.type == self.mt5.POSITION_TYPE_BUY:
                be_sl = pos.price_open + snapshot.atr * 0.10
                if pos.sl <= 0 or be_sl > pos.sl:
                    self._modify_position(pos, sl=be_sl, tp=pos.tp)
            else:
                be_sl = pos.price_open - snapshot.atr * 0.10
                if pos.sl <= 0 or be_sl < pos.sl:
                    self._modify_position(pos, sl=be_sl, tp=pos.tp)

        if profit_move < snapshot.atr * 1.5:
            return

        if pos.type == self.mt5.POSITION_TYPE_BUY:
            new_sl = max(pos.sl, current_price - snapshot.atr * 0.5)
            if pos.sl <= 0 or new_sl > pos.sl:
                self._modify_position(pos, sl=self._round_to_digits(new_sl, self._digits()), tp=pos.tp)
        else:
            new_sl = min(pos.sl if pos.sl > 0 else current_price + snapshot.atr * 100.0, current_price + snapshot.atr * 0.5)
            if pos.sl <= 0 or new_sl < pos.sl:
                self._modify_position(pos, sl=self._round_to_digits(new_sl, self._digits()), tp=pos.tp)

    def _enter(self, snapshot: MarketSnapshot) -> None:
        direction = snapshot.breakout_signal
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
            "ENTRY %s volume=%.2f price=%.2f sl=%.2f tp=%.2f equity=%.2f score=%.2f",
            direction,
            volume,
            price,
            sl,
            tp,
            self._get_equity(),
            snapshot.score,
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
        self._save_state()

    def _build_sl_tp(self, direction: str, price: float, snapshot: MarketSnapshot) -> Tuple[float, float]:
        asia_width = max(snapshot.asia_width, self._point() * 10)
        sl_buffer = max(snapshot.atr * self.config.stop_buffer_atr, asia_width * 0.25, self._point() * 15)
        risk_distance = sl_buffer
        tp_distance = risk_distance * self.config.reward_multiple
        digits = self._digits()

        if direction == "BUY":
            sl = min(snapshot.asia_low - sl_buffer, price - risk_distance)
            tp = price + tp_distance
        else:
            sl = max(snapshot.asia_high + sl_buffer, price + risk_distance)
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
            "comment": "xauusd-ep1-live" if self.config.live else "xauusd-ep1-dryrun",
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
            "comment": "ep1-trail-adjust",
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
                "comment": "ep1-close-all",
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
        info = self.mt5.symbol_info(self.symbol)
        if info is None:
            raise RuntimeError("symbol_info unavailable")
        return float(getattr(info, "point", 0.0) or 0.0)

    def _digits(self) -> int:
        info = self.mt5.symbol_info(self.symbol)
        if info is None:
            raise RuntimeError("symbol_info unavailable")
        return int(getattr(info, "digits", 0) or 0)

    def _spread_points(self) -> float:
        tick = self.mt5.symbol_info_tick(self.symbol)
        if tick is None:
            raise RuntimeError(f"symbol_info_tick failed: {self.mt5.last_error()}")
        ask = float(getattr(tick, "ask", 0.0) or getattr(tick, "last", 0.0) or 0.0)
        bid = float(getattr(tick, "bid", 0.0) or getattr(tick, "last", 0.0) or 0.0)
        if ask <= 0 or bid <= 0:
            raise RuntimeError(f"Invalid bid/ask: ask={ask} bid={bid}")
        point = self._point()
        if point <= 0:
            raise RuntimeError("point unavailable")
        return abs(ask - bid) / point

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

    def _session_allowed(self, bar_time: dt.datetime, start: int, end: int) -> bool:
        hour = bar_time.hour
        if start == end:
            return True
        if start < end:
            return start <= hour < end
        return hour >= start or hour < end

    def _trade_window_allowed(self, bar_time: dt.datetime) -> bool:
        return self._session_allowed(bar_time, self.config.london_start_utc, self.config.london_end_utc)

    def _cooldown_ok(self, bar_time: dt.datetime) -> bool:
        if self.state.last_trade_bar_time is None:
            return True
        try:
            last_dt = dt.datetime.fromisoformat(str(self.state.last_trade_bar_time))
        except Exception:
            return True
        delta_bars = (bar_time - last_dt).total_seconds() / max(self.config.loop_seconds, 1)
        return delta_bars >= float(self.config.cooldown_bars_after_trade)

    def _max_hold_exceeded(self, pos: PositionState, bar_time: dt.datetime) -> bool:
        if pos.time_open is None:
            return False
        opened = dt.datetime.fromtimestamp(int(pos.time_open))
        opened = self._adjust_time(opened)
        held_minutes = (bar_time - opened).total_seconds() / 60.0
        return held_minutes >= float(self.config.max_hold_minutes)

    def _sync_closed_trades(self) -> None:
        from_time = dt.datetime.now() - dt.timedelta(days=30)
        to_time = dt.datetime.now()
        try:
            deals = self.mt5.history_deals_get(from_time, to_time)
        except Exception:
            return
        if not deals:
            return
        latest_seen = self.state.last_processed_deal_time
        for deal in list(deals):
            deal_time = getattr(deal, "time", None)
            if deal_time is None:
                continue
            deal_iso = dt.datetime.fromtimestamp(int(deal_time)).isoformat()
            if latest_seen is not None and deal_iso <= latest_seen:
                continue
            profit = float(getattr(deal, "profit", 0.0)) + float(getattr(deal, "commission", 0.0)) + float(getattr(deal, "swap", 0.0))
            self.state.last_close_profit = profit
            self.state.last_processed_deal_time = deal_iso
            latest_seen = deal_iso
            if profit < 0:
                self.state.consecutive_losses += 1
            elif profit > 0:
                self.state.consecutive_losses = 0
            self._log(
                "Closed deal sync | time=%s profit=%.2f consecutive_losses=%d",
                deal_iso,
                profit,
                self.state.consecutive_losses,
            )
        self._save_state()

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
    parser = argparse.ArgumentParser(description="XAUUSD Asia-London breakout MT5 Python strategy.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M5")
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
    parser.add_argument("--asia-start-utc", type=int, default=0)
    parser.add_argument("--asia-end-utc", type=int, default=6)
    parser.add_argument("--london-start-utc", type=int, default=7)
    parser.add_argument("--london-end-utc", type=int, default=16)
    parser.add_argument("--time-offset-hours", type=int, default=0, help="Shift MT5 server bars to UTC.")
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--breakout-buffer-atr", type=float, default=0.15)
    parser.add_argument("--stop-buffer-atr", type=float, default=1.0)
    parser.add_argument("--reward-multiple", type=float, default=2.2)
    parser.add_argument("--max-spread-points", type=float, default=120.0)
    parser.add_argument("--max-trades-per-day", type=int, default=2)
    parser.add_argument("--max-consecutive-losses", type=int, default=2)
    parser.add_argument("--cooldown-bars-after-trade", type=int, default=3)
    parser.add_argument("--max-hold-minutes", type=int, default=240)
    parser.add_argument("--loop-seconds", type=int, default=10)
    parser.add_argument("--lookback-bars", type=int, default=300)
    parser.add_argument("--htf-lookback-bars", type=int, default=400)
    parser.add_argument("--htf-fast-sma", type=int, default=50)
    parser.add_argument("--htf-slow-sma", type=int, default=200)
    parser.add_argument("--min-asia-range-atr", type=float, default=0.55)
    parser.add_argument("--max-asia-range-atr", type=float, default=3.5)
    parser.add_argument("--state-path", default=STATE_PATH_DEFAULT)
    parser.add_argument("--log-file", default=LOG_PATH_DEFAULT)
    parser.add_argument(
        "--terminal-path",
        default=r"C:\Program Files\MetaTrader 5\terminal64.exe",
        help="Windows terminal path inside the Wine prefix. Set empty to skip explicit initialize(path=...).",
    )
    parser.add_argument("--deviation", type=int, default=30)
    parser.add_argument("--magic", type=int, default=204495)
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
        asia_start_utc=int(args.asia_start_utc),
        asia_end_utc=int(args.asia_end_utc),
        london_start_utc=int(args.london_start_utc),
        london_end_utc=int(args.london_end_utc),
        time_offset_hours=int(args.time_offset_hours),
        atr_period=int(args.atr_period),
        breakout_buffer_atr=float(args.breakout_buffer_atr),
        stop_buffer_atr=float(args.stop_buffer_atr),
        reward_multiple=float(args.reward_multiple),
        max_spread_points=float(args.max_spread_points),
        max_trades_per_day=int(args.max_trades_per_day),
        max_consecutive_losses=int(args.max_consecutive_losses),
        cooldown_bars_after_trade=int(args.cooldown_bars_after_trade),
        max_hold_minutes=int(args.max_hold_minutes),
        loop_seconds=int(args.loop_seconds),
        lookback_bars=int(args.lookback_bars),
        htf_lookback_bars=int(args.htf_lookback_bars),
        htf_fast_sma=int(args.htf_fast_sma),
        htf_slow_sma=int(args.htf_slow_sma),
        min_asia_range_atr=float(args.min_asia_range_atr),
        max_asia_range_atr=float(args.max_asia_range_atr),
        state_path=args.state_path,
        log_file=args.log_file,
        terminal_path=terminal_path,
        deviation=int(args.deviation),
        magic=int(args.magic),
        log_level=args.log_level,
    )

    strategy = XAUUSDAsiaLondonBreakoutStrategy(config)
    strategy.run()


if __name__ == "__main__":
    main()
