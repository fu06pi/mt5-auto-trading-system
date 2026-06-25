#!/usr/bin/env python3.14
"""Asian session pivot-stall reversal strategy for XAUUSD.

Detects rolling pivot highs/lows in Asian session, waits for stall
bars (tight range + declining volume), then fades the reversal.

Risk controls: daily DD 3%, total DD 10%, profit target 5%.
One position at a time. MT5 bridge-friendly.
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


LOGGER = logging.getLogger("xauusd_asian_reversal")
STATE_PATH_DEFAULT = "/home/chain4655/Documents/Sample/Python/xauusd_asian_reversal_state.json"
LOG_PATH_DEFAULT = "/home/chain4655/Documents/Sample/Python/xauusd_asian_reversal_strategy.log"
REALIZED_PNL_NOISE_USD = 4.0


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
    atr_period: int
    stop_atr: float
    take_profit_atr: float
    pivot_lookback: int
    stall_bars: int
    stall_range_atr: float
    stall_vol_ratio: float
    reversal_atr: float
    session_start_hour: int
    session_end_hour: int
    max_trades_per_day: int
    max_hold_minutes: int
    max_consecutive_losses: int
    loss_cooldown_minutes: int
    max_concentration_share: float
    min_positive_days_for_concentration: int
    cooldown_bars_after_trade: int
    startup_warmup_bars: int
    loop_seconds: int
    lookback_bars: int
    max_spread_points: float
    state_path: str
    log_file: str
    deviation: int
    magic: int
    log_level: str
    max_leverage: float = 5.0
    order_comment: str = ""


@dataclasses.dataclass
class MarketSnapshot:
    bar_time: dt.datetime
    close: float
    high: float
    low: float
    atr: float
    session: str
    spread_points: float


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
    positive_days_profit: float = 0.0
    positive_days_count: int = 0
    best_day_profit: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    loss_cooldown_until: Optional[str] = None
    traded_bar_indices: List[int] = dataclasses.field(default_factory=list)
    last_processed_deal_time: Optional[str] = None
    last_close_profit: float = 0.0
    paused_reason: str = ""
    paused: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StrategyState:
        fields = dataclasses.fields(cls)
        kwargs = {}
        for i, f in enumerate(fields):
            default = f.default if isinstance(f.default, (int, float, str, bool, type(None))) else []
            kwargs[f.name] = data.get(f.name, default)
        return cls(**kwargs)


class XAUUSDAsianReversalStrategy:

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self.mt5 = MetaTrader5(host=config.host, port=config.port)
        self.state = StrategyState()
        self._bar_time: Optional[dt.datetime] = None
        self._last_bar_time: Optional[dt.datetime] = None
        self._pivot_high_idx: int = -1
        self._pivot_low_idx: int = -1
        self._last_trade_day: Optional[str] = None
        self._connect_fail_streak = 0
        self._last_reconnect_ts = 0.0

        # Resolve timeframe seconds
        tf_map = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600}
        self._tf_seconds = next((v for k, v in tf_map.items() if k.upper() == config.timeframe.upper()), 60)
        self._loop_start = time.time()

    def _mt5_timestamp_to_chart_time(self, timestamp: int) -> dt.datetime:
        """Convert MT5 epoch seconds to naive broker chart/server time."""
        return dt.datetime.fromtimestamp(int(timestamp), dt.UTC).replace(tzinfo=None)

    def _chart_now(self) -> dt.datetime:
        """Return naive MT5 chart/server time for cooldown/state comparisons."""
        try:
            tick = self.mt5.symbol_info_tick(self.config.symbol)
            tick_time = int(getattr(tick, "time", 0) or 0) if tick is not None else 0
            if tick_time > 0:
                return self._mt5_timestamp_to_chart_time(tick_time)
        except Exception:
            pass
        return dt.datetime.now(dt.UTC).replace(tzinfo=None)

    # ── Pivot Detection ──────────────────────────────────────────────

    def _is_pivot_high(self, bars: List, i: int) -> bool:
        lb = self.config.pivot_lookback
        if i < lb or i >= len(bars) - lb:
            return False
        for o in range(1, lb + 1):
            if bars[i].high <= bars[i - o].high or bars[i].high <= bars[i + o].high:
                return False
        return True

    def _is_pivot_low(self, bars: List, i: int) -> bool:
        lb = self.config.pivot_lookback
        if i < lb or i >= len(bars) - lb:
            return False
        for o in range(1, lb + 1):
            if bars[i].low >= bars[i - o].low or bars[i].low >= bars[i + o].low:
                return False
        return True

    # ── Core Logic ───────────────────────────────────────────────────

    def _build_snapshot(self) -> MarketSnapshot:
        bars = self._get_bars(self.config.lookback_bars)
        if len(bars) < self.config.atr_period + self.config.pivot_lookback + 5:
            raise RuntimeError("Not enough bars")
        atr = self._atr(bars)
        return MarketSnapshot(
            bar_time=bars[-1].time,
            close=bars[-1].close,
            high=bars[-1].high,
            low=bars[-1].low,
            atr=atr,
            session=self._session_of(bars[-1].time),
            spread_points=self._spread(),
        )

    def _handle_bar(self, snapshot: MarketSnapshot) -> None:
        if snapshot.session not in ("asia",):
            self._pivot_high_idx = self._pivot_low_idx = -1
            return

        bars = self._get_bars(self.config.lookback_bars)
        if len(bars) < self.config.pivot_lookback + self.config.stall_bars + 10:
            return

        bar_idx = len(bars) - 1

        # Update pivots
        for i in range(max(0, bar_idx - self.config.pivot_lookback - 2), bar_idx + 1):
            if self._is_pivot_high(bars, i):
                self._pivot_high_idx = i
            if self._is_pivot_low(bars, i):
                self._pivot_low_idx = i

        if self._last_trade_day == snapshot.bar_time.date().isoformat():
            return
        if self.state.paused:
            return
        if self.state.trades_today >= self.config.max_trades_per_day:
            return
        if self._loss_cooldown_active():
            return

        # Try high stall (SELL)
        side, pivot_price, pivot_type = None, 0.0, ""
        if self._pivot_high_idx >= 0 and bar_idx > self._pivot_high_idx + self.config.stall_bars:
            stall_slice = bars[self._pivot_high_idx + 1:bar_idx + 1]
            if len(stall_slice) >= self.config.stall_bars:
                stall_used = stall_slice[-self.config.stall_bars:]
                stall_ranges = [b.high - b.low for b in stall_used]
                avg_range_atr = statistics.fmean(stall_ranges) / max(snapshot.atr, 0.01)
                if avg_range_atr <= self.config.stall_range_atr:
                    pivot_vol = bars[self._pivot_high_idx].tick_volume
                    stall_vol = statistics.fmean([b.tick_volume for b in stall_used])
                    vol_ratio = stall_vol / max(pivot_vol, 1)
                    if vol_ratio <= self.config.stall_vol_ratio:
                        curr = bars[-1]
                        if (curr.close < bars[self._pivot_high_idx].high
                                - self.config.reversal_atr * snapshot.atr
                                and curr.close < curr.open):
                            side, pivot_price, pivot_type = "SELL", bars[self._pivot_high_idx].high, "high_stall"

        # Try low stall (BUY)
        if side is None and self._pivot_low_idx >= 0 and bar_idx > self._pivot_low_idx + self.config.stall_bars:
            stall_slice = bars[self._pivot_low_idx + 1:bar_idx + 1]
            if len(stall_slice) >= self.config.stall_bars:
                stall_used = stall_slice[-self.config.stall_bars:]
                stall_ranges = [b.high - b.low for b in stall_used]
                avg_range_atr = statistics.fmean(stall_ranges) / max(snapshot.atr, 0.01)
                if avg_range_atr <= self.config.stall_range_atr:
                    pivot_vol = bars[self._pivot_low_idx].tick_volume
                    stall_vol = statistics.fmean([b.tick_volume for b in stall_used])
                    vol_ratio = stall_vol / max(pivot_vol, 1)
                    if vol_ratio <= self.config.stall_vol_ratio:
                        curr = bars[-1]
                        if (curr.close > bars[self._pivot_low_idx].low
                                + self.config.reversal_atr * snapshot.atr
                                and curr.close > curr.open):
                            side, pivot_price, pivot_type = "BUY", bars[self._pivot_low_idx].low, "low_stall"

        if side is None:
            return

        self._enter(side, snapshot)
        self._last_trade_day = snapshot.bar_time.date().isoformat()

    def _enter(self, side: str, snapshot: MarketSnapshot) -> None:
        self._ensure_mt5()
        entry_price = snapshot.close
        atr = snapshot.atr
        if side == "BUY":
            sl = entry_price - self.config.stop_atr * atr
            tp = entry_price + self.config.take_profit_atr * atr
        else:
            sl = entry_price + self.config.stop_atr * atr
            tp = entry_price - self.config.take_profit_atr * atr

        volume = self._size_position(side, entry_price, sl)

        if not self.config.live:
            LOGGER.info("DRY-RUN %s %.2f sl=%.2f tp=%.2f vol=%.2f atr=%.2f",
                        side, entry_price, sl, tp, volume, atr)
            self.state.trades_today += 1
            self._save_state()
            return

        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": self.config.symbol,
            "volume": volume,
            "type": self.mt5.ORDER_TYPE_BUY if side == "BUY" else self.mt5.ORDER_TYPE_SELL,
            "price": entry_price,
            "sl": sl,
            "tp": tp,
            "deviation": self.config.deviation,
            "magic": self.config.magic,
            "comment": self.config.order_comment or "asrv",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        try:
            result = self.mt5.order_send(request)
        except Exception as exc:
            LOGGER.error("order_send exception: %s", exc)
            return

        if result is None:
            LOGGER.error("order_send returned None: %s", self.mt5.last_error())
            return

        retcode = getattr(result, "retcode", -1)
        if retcode != self.mt5.TRADE_RETCODE_DONE:
            LOGGER.error("order_send failed: retcode=%s error=%s", retcode, self.mt5.last_error())
            return

        LOGGER.info("ORDER FILLED %s %.2f sl=%.2f tp=%.2f vol=%.2f atr=%.2f ticket=%s",
                    side, entry_price, sl, tp, volume, atr, getattr(result, "order", "?"))
        self.state.trades_today += 1
        self._save_state()

    def _size_position(self, side: str, entry: float, sl: float) -> float:
        equity = self._equity()
        risk_amount = equity * self.config.risk_pct
        risk_points = abs(entry - sl)
        raw_vol = risk_amount / max(risk_points, 0.01) / 100.0
        return max(0.01, min(raw_vol, self.config.max_lots))

    # ── Risk Controls ────────────────────────────────────────────────

    def _risk_guard(self, snapshot: MarketSnapshot) -> None:
        if self.state.paused:
            return

        equity = self._equity()
        day_start = self.state.day_start_equity or equity
        initial = self.state.initial_equity or equity

        # Daily DD
        day_dd = (day_start - equity) / max(day_start, 1)
        if day_dd >= self.config.daily_dd_limit:
            self._pause(f"daily_dd {day_dd:.1%}")
            return

        # Total DD
        total_dd = (initial - equity) / max(initial, 1)
        if total_dd >= self.config.total_dd_limit:
            self._pause(f"total_dd {total_dd:.1%}")
            return

        # Profit target
        total_profit = (equity - initial) / max(initial, 1)
        if total_profit >= self.config.profit_target:
            self._pause(f"profit_target {total_profit:.1%}")
            return

        # Concentration guard
        self._check_concentration(equity)

    def _check_concentration(self, equity: float) -> None:
        today_pnl = equity - (self.state.day_start_equity or equity)
        if today_pnl <= 0:
            return
        if self.state.positive_days_count < self.config.min_positive_days_for_concentration:
            return
        max_allowed = self.state.positive_days_profit * self.config.max_concentration_share
        if today_pnl > max_allowed and self.state.positive_days_count > 0:
            self._pause("concentration_guard")

    def _pause(self, reason: str) -> None:
        self.state.paused = True
        self.state.paused_reason = reason
        LOGGER.warning("PAUSED: %s", reason)
        self.close_all_positions()
        self._save_state()

    def _loss_cooldown_active(self) -> bool:
        if self.state.loss_cooldown_until is None:
            return False
        try:
            until = dt.datetime.fromisoformat(str(self.state.loss_cooldown_until))
        except ValueError:
            self.state.loss_cooldown_until = None
            self._save_state()
            return False
        current = self._chart_now()
        if current < until:
            return True
        self.state.loss_cooldown_until = None
        self._save_state()
        return False

    # ── Position Management ──────────────────────────────────────────

    def close_all_positions(self) -> bool:
        self._ensure_mt5()
        positions = self._positions()
        if not positions:
            return True
        ok = True
        for pos in positions:
            try:
                price = self.mt5.symbol_info_tick(self.config.symbol).bid if pos.type == 0 else self.mt5.symbol_info_tick(self.config.symbol).ask
                req = {
                    "action": self.mt5.TRADE_ACTION_DEAL,
                    "symbol": self.config.symbol,
                    "volume": pos.volume,
                    "type": self.mt5.ORDER_TYPE_SELL if pos.type == self.mt5.POSITION_TYPE_BUY else self.mt5.ORDER_TYPE_BUY,
                    "position": pos.ticket,
                    "price": price,
                    "deviation": self.config.deviation,
                    "magic": self.config.magic,
                    "type_time": self.mt5.ORDER_TIME_GTC,
                    "type_filling": self.mt5.ORDER_FILLING_IOC,
                }
                result = self.mt5.order_send(req)
                if result is None or getattr(result, "retcode", -1) != self.mt5.TRADE_RETCODE_DONE:
                    LOGGER.error("close failed ticket=%s: %s", pos.ticket, self.mt5.last_error())
                    ok = False
            except Exception as exc:
                LOGGER.error("close exception ticket=%s: %s", pos.ticket, exc)
                ok = False
        return ok

    def _sync_closed_trades(self) -> None:
        self._ensure_mt5()
        to_time = self._chart_now()
        from_time = to_time - dt.timedelta(days=30)
        try:
            deals = self.mt5.history_deals_get(from_time, to_time)
        except Exception:
            return
        if not deals:
            return
        latest_seen = self.state.last_processed_deal_time
        ordered = sorted(deals, key=lambda d: (int(getattr(d, "time", 0) or 0), int(getattr(d, "ticket", 0) or 0)))
        for deal in ordered:
            deal_time = getattr(deal, "time", None)
            if deal_time is None:
                continue
            deal_dt = self._mt5_timestamp_to_chart_time(int(deal_time))
            deal_iso = deal_dt.isoformat()
            if latest_seen is not None and deal_iso <= latest_seen:
                continue
            if not self._is_owned_closing_deal(deal):
                continue
            profit = float(getattr(deal, "profit", 0) or 0)
            self.state.last_close_profit = profit
            self.state.last_processed_deal_time = deal_iso
            latest_seen = deal_iso
            if abs(profit) < REALIZED_PNL_NOISE_USD:
                continue
            if profit < 0:
                self.state.consecutive_losses += 1
                if self.state.consecutive_losses >= self.config.max_consecutive_losses:
                    until = (deal_dt + dt.timedelta(minutes=self.config.loss_cooldown_minutes)).isoformat()
                    self.state.loss_cooldown_until = until
                    LOGGER.warning("Loss cooldown activated: %s", until)
            elif profit > 0:
                self.state.consecutive_losses = 0
                self.state.loss_cooldown_until = None
            LOGGER.info("Closed deal sync | profit=%.2f consecutive_losses=%d", profit, self.state.consecutive_losses)
        self._save_state()

    def _is_owned_closing_deal(self, deal: Any) -> bool:
        magic = int(getattr(deal, "magic", 0) or 0)
        if magic != self.config.magic:
            return False
        entry = getattr(deal, "entry", None)
        close_entries = {getattr(self.mt5, "DEAL_ENTRY_OUT", 1), getattr(self.mt5, "DEAL_ENTRY_OUT_BY", 3)}
        return entry is None or int(entry) in close_entries

    # ── MT5 Bridge Workaround ─────────────────────────────────────────

    pymt5linux_bridge_broken_reinit: bool = False

    def _recreate_mt5_client(self, reason: str) -> None:
        self._connect_fail_streak += 1
        LOGGER.warning("Recreating MT5 client (streak=%s): %s", self._connect_fail_streak, reason)
        now = time.time()
        if now - self._last_reconnect_ts < 5:
            time.sleep(2)
        self._last_reconnect_ts = now
        try:
            self.mt5.shutdown()
        except Exception:
            pass
        self.mt5 = MetaTrader5(host=self.config.host, port=self.config.port)
        ok = self.mt5.initialize()
        if not ok:
            raise RuntimeError(f"MT5 reinitialize failed: {self.mt5.last_error()}")
        if not self.mt5.symbol_select(self.config.symbol, True):
            raise RuntimeError(f"symbol_select failed for {self.config.symbol}: {self.mt5.last_error()}")
        self._connect_fail_streak = 0

    def _ensure_mt5(self) -> None:
        try:
            ok = self.mt5.initialize()
            if not ok:
                LOGGER.debug("mt5.initialize() returned False: %s", self.mt5.last_error())
                self._recreate_mt5_client("initialize returned False")
                return
            if not self.mt5.symbol_select(self.config.symbol, True):
                self._recreate_mt5_client(f"symbol_select failed: {self.mt5.last_error()}")
                return
            self.pymt5linux_bridge_broken_reinit = True
        except Exception as exc:
            self._recreate_mt5_client(f"_ensure_mt5 exception: {exc}")

    # ── Data Helpers ─────────────────────────────────────────────────

    def _get_bars(self, count: int) -> List:
        tf = self._timeframe_constant()
        for attempt in range(3):
            self._ensure_mt5()
            rates = self.mt5.copy_rates_from_pos(self.config.symbol, tf, 0, count)
            if rates is not None and len(rates) > 0:
                return [Bar(r) for r in rates]
            LOGGER.warning(
                "copy_rates_from_pos unavailable (attempt %s/3): %s",
                attempt + 1,
                self.mt5.last_error(),
            )
            if attempt < 2:
                self._recreate_mt5_client("copy_rates_from_pos unavailable")
                time.sleep(1)
        return []

    def _timeframe_constant(self) -> int:
        tf_map = {"M1": self.mt5.TIMEFRAME_M1, "M5": self.mt5.TIMEFRAME_M5,
                  "M15": self.mt5.TIMEFRAME_M15, "M30": self.mt5.TIMEFRAME_M30,
                  "H1": self.mt5.TIMEFRAME_H1}
        return tf_map.get(self.config.timeframe.upper(), self.mt5.TIMEFRAME_M1)

    def _atr(self, bars: List) -> float:
        if len(bars) < self.config.atr_period + 1:
            return 0.01
        trs = []
        for i in range(-self.config.atr_period, 0):
            c, p = bars[i], bars[i - 1]
            trs.append(max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close)))
        return max(statistics.fmean(trs), 0.01)

    def _spread(self) -> float:
        self._ensure_mt5()
        try:
            info = self.mt5.symbol_info(self.config.symbol)
            if info:
                return float((getattr(info, "spread", 0) or 0))
        except Exception:
            pass
        return 0.0

    def _session_of(self, ts: dt.datetime) -> str:
        h = ts.hour
        if self.config.session_start_hour <= h < self.config.session_end_hour:
            return "asia"
        return "other"

    def _positions(self) -> List[PositionState]:
        self._ensure_mt5()
        raw = self.mt5.positions_get(symbol=self.config.symbol)
        if raw is None:
            return []
        out = []
        for pos in raw:
            if int(getattr(pos, "magic", 0)) == self.config.magic:
                out.append(PositionState(
                    ticket=int(getattr(pos, "ticket", 0)),
                    symbol=str(getattr(pos, "symbol", self.config.symbol)),
                    magic=self.config.magic,
                    type=int(getattr(pos, "type", 0)),
                    volume=float(getattr(pos, "volume", 0.0)),
                    price_open=float(getattr(pos, "price_open", 0.0)),
                    sl=float(getattr(pos, "sl", 0.0)),
                    tp=float(getattr(pos, "tp", 0.0)),
                    profit=float(getattr(pos, "profit", 0.0)),
                    time_open=int(getattr(pos, "time", 0)) if getattr(pos, "time", None) is not None else None,
                ))
        return out

    def _equity(self) -> float:
        self._ensure_mt5()
        try:
            info = self.mt5.account_info()
            if info:
                return float(getattr(info, "equity", 0) or 0)
        except Exception:
            pass
        return self.state.day_start_equity or self.config.start_equity

    # ── Lifecycle ────────────────────────────────────────────────────

    def _connect(self) -> None:
        ok = self.mt5.initialize()
        if not ok:
            raise RuntimeError(f"MT5 init failed: {self.mt5.last_error()}")
        self.mt5.symbol_select(self.config.symbol, True)

    def _shutdown(self) -> None:
        try:
            self.mt5.shutdown()
        except Exception:
            pass

    def _ensure_day_context(self) -> None:
        today = self._chart_now().date().isoformat()
        if self.state.current_day != today:
            equity = self._equity()
            self.state.current_day = today
            self.state.day_start_equity = equity
            self.state.trades_today = 0
            if self.state.initial_equity is None:
                self.state.initial_equity = equity
            self.state.last_equity = equity
            self.state.max_equity_seen = max(self.state.max_equity_seen or equity, equity)
            self._last_trade_day = None
            self._last_bar_time = None
            LOGGER.info("New day: %s equity=%.2f", today, equity)
            self._save_state()

    def _save_state(self) -> None:
        try:
            path = Path(self.config.state_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.state.to_dict(), indent=2))
        except Exception as exc:
            LOGGER.error("save_state failed: %s", exc)

    def _load_state(self) -> None:
        try:
            path = Path(self.config.state_path)
            if path.exists():
                data = json.loads(path.read_text())
                self.state = StrategyState.from_dict(data)
                LOGGER.info("State loaded: %d trades today, %d consecutive losses",
                            self.state.trades_today, self.state.consecutive_losses)
        except Exception as exc:
            LOGGER.warning("load_state failed (fresh start): %s", exc)

    def _seed_equity(self) -> None:
        if self.state.initial_equity is None:
            self.state.initial_equity = self._equity()
            self.state.day_start_equity = self.state.initial_equity
            self.state.max_equity_seen = self.state.initial_equity

    def run(self) -> None:
        LOGGER.info("Starting XAUUSD Asian Reversal Strategy (magic=%s, %s, %s)",
                    self.config.magic, self.config.symbol, self.config.timeframe)
        self._connect()
        self._load_state()
        self._seed_equity()

        try:
            while True:
                self._loop_start = time.time()
                try:
                    snapshot = self._build_snapshot()
                except Exception as exc:
                    LOGGER.warning("build_snapshot failed: %s", exc)
                    time.sleep(self.config.loop_seconds)
                    continue

                self._ensure_day_context()
                self._risk_guard(snapshot)
                self._sync_closed_trades()

                cur_bar = snapshot.bar_time
                if self._bar_time is None or cur_bar > self._bar_time:
                    self._last_bar_time = self._bar_time
                    self._bar_time = cur_bar
                    if self._last_bar_time is not None:
                        self._handle_bar(snapshot)

                positions = self._positions()

                LOGGER.info(
                    "Bar %s | close=%.2f atr=%.2f session=%s spread=%.1f positions=%d trades_today=%d losses=%d",
                    cur_bar.strftime("%H:%M"), snapshot.close, snapshot.atr,
                    snapshot.session, snapshot.spread_points,
                    len(positions), self.state.trades_today, self.state.consecutive_losses,
                )

                elapsed = time.time() - self._loop_start
                sleep_time = max(0.1, self.config.loop_seconds - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            LOGGER.info("Shutdown by signal")
        finally:
            self._save_state()
            self._shutdown()


class Bar:
    __slots__ = ("time", "open", "high", "low", "close", "tick_volume")
    def __init__(self, row: Any) -> None:
        self.time = dt.datetime.fromtimestamp(int(row["time"]), tz=dt.UTC).replace(tzinfo=None)
        self.open = float(row["open"])
        self.high = float(row["high"])
        self.low = float(row["low"])
        self.close = float(row["close"])
        self.tick_volume = float(row["tick_volume"])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="XAUUSD Asian session pivot-stall reversal strategy")
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--timeframe", default="M1")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18812)
    p.add_argument("--live", action="store_true")
    p.add_argument("--start-equity", type=float, default=10000.0)
    p.add_argument("--daily-dd-limit", type=float, default=0.03)
    p.add_argument("--total-dd-limit", type=float, default=0.10)
    p.add_argument("--profit-target", type=float, default=0.05)
    p.add_argument("--risk-pct", type=float, default=0.0035)
    p.add_argument("--max-lots", type=float, default=2.0)
    p.add_argument("--atr-period", type=int, default=14)
    p.add_argument("--stop-atr", type=float, default=1.5)
    p.add_argument("--take-profit-atr", type=float, default=6.0)
    p.add_argument("--pivot-lookback", type=int, default=4)
    p.add_argument("--stall-bars", type=int, default=3)
    p.add_argument("--stall-range-atr", type=float, default=0.5)
    p.add_argument("--stall-vol-ratio", type=float, default=0.9)
    p.add_argument("--reversal-atr", type=float, default=0.25)
    p.add_argument("--session-start-hour", type=int, default=0)
    p.add_argument("--session-end-hour", type=int, default=7)
    p.add_argument("--max-trades-per-day", type=int, default=1)
    p.add_argument("--max-hold-minutes", type=int, default=48)
    p.add_argument("--max-consecutive-losses", type=int, default=3)
    p.add_argument("--loss-cooldown-minutes", type=int, default=120)
    p.add_argument("--max-concentration-share", type=float, default=0.30)
    p.add_argument("--min-positive-days-for-concentration", type=int, default=5)
    p.add_argument("--cooldown-bars-after-trade", type=int, default=5)
    p.add_argument("--startup-warmup-bars", type=int, default=50)
    p.add_argument("--loop-seconds", type=float, default=5.0)
    p.add_argument("--lookback-bars", type=int, default=200)
    p.add_argument("--max-spread-points", type=float, default=50.0)
    p.add_argument("--state-path", default=STATE_PATH_DEFAULT)
    p.add_argument("--log-file", default=LOG_PATH_DEFAULT)
    p.add_argument("--deviation", type=int, default=10)
    p.add_argument("--magic", type=int, default=204574)
    p.add_argument("--max-leverage", type=float, default=5.0)
    p.add_argument("--order-comment", default="")
    p.add_argument("--log-level", default="INFO")
    return p


def main() -> None:
    args = build_parser().parse_args()
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_handlers: List[logging.Handler] = [logging.StreamHandler()]
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=log_handlers,
        force=True,
    )
    config = StrategyConfig(
        symbol=args.symbol, timeframe=args.timeframe, host=args.host, port=args.port,
        live=bool(args.live), start_equity=float(args.start_equity),
        daily_dd_limit=float(args.daily_dd_limit), total_dd_limit=float(args.total_dd_limit),
        profit_target=float(args.profit_target), risk_pct=float(args.risk_pct),
        max_lots=float(args.max_lots), atr_period=int(args.atr_period),
        stop_atr=float(args.stop_atr), take_profit_atr=float(args.take_profit_atr),
        pivot_lookback=int(args.pivot_lookback), stall_bars=int(args.stall_bars),
        stall_range_atr=float(args.stall_range_atr), stall_vol_ratio=float(args.stall_vol_ratio),
        reversal_atr=float(args.reversal_atr),
        session_start_hour=int(args.session_start_hour), session_end_hour=int(args.session_end_hour),
        max_trades_per_day=int(args.max_trades_per_day), max_hold_minutes=int(args.max_hold_minutes),
        max_consecutive_losses=int(args.max_consecutive_losses),
        loss_cooldown_minutes=int(args.loss_cooldown_minutes),
        max_concentration_share=float(args.max_concentration_share),
        min_positive_days_for_concentration=int(args.min_positive_days_for_concentration),
        cooldown_bars_after_trade=int(args.cooldown_bars_after_trade),
        startup_warmup_bars=int(args.startup_warmup_bars), loop_seconds=float(args.loop_seconds),
        lookback_bars=int(args.lookback_bars), max_spread_points=float(args.max_spread_points),
        state_path=args.state_path, log_file=args.log_file, deviation=int(args.deviation),
        magic=int(args.magic), max_leverage=float(args.max_leverage),
        order_comment=str(args.order_comment), log_level=args.log_level,
    )
    strategy = XAUUSDAsianReversalStrategy(config)
    strategy.run()


if __name__ == "__main__":
    main()
