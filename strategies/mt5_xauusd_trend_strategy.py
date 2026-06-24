#!/usr/bin/env python3.14
"""XAUUSD trend-following MT5 strategy with hard risk controls.

Rules implemented:
- Starting capital reference: 10000
- Max daily drawdown: 3%
- Max total drawdown: 10%
- Profit target: 5%
- Best Day concentration guard: track positive-day profits and avoid letting
  one day dominate the cumulative positive-day profit bucket.

Design goals:
- Trade XAUUSD only
- Trend-following bias using SMA structure, ATR, momentum, and breakout filter
- One-symbol, one-account, one-position-at-a-time
- Bridge-friendly for pymt5linux on Wine-hosted MT5
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
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from pymt5linux import MetaTrader5


LOGGER = logging.getLogger("xauusd_trend_strategy")
STATE_PATH_DEFAULT = "/home/chain4655/Documents/Sample/Python/xauusd_trend_state.json"
LOG_PATH_DEFAULT = "/home/chain4655/Documents/Sample/Python/xauusd_trend_strategy.log"
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
    max_lots_per_order: float
    max_leverage: float
    fast_sma: int
    slow_sma: int
    htf_timeframe: str
    htf_fast_sma: int
    htf_slow_sma: int
    trend_threshold: float
    htf_comp_momentum_threshold: float
    htf_momentum_bias_weight: float
    momentum_score_weight: float
    atr_period: int
    breakout_lookback: int
    enable_false_breakout_reversal: bool
    false_breakout_direction: str
    false_breakout_lookback: int
    false_breakout_min_atr: float
    false_breakout_close_back_atr: float
    false_breakout_wick_ratio: float
    stop_atr: float
    reward_multiple: float
    trail_trigger_atr: float
    trail_lock_atr: float
    trail_stable_minutes: int
    trail_same_direction_cooldown_minutes: int
    break_even_atr: float
    break_even_lock_atr: float
    fee_cover_price_offset: float
    session_start_utc: int
    session_end_utc: int
    max_spread_points: float
    max_trades_per_day: int
    max_consecutive_losses: int
    loss_cooldown_losses: int
    loss_cooldown_minutes: int
    auto_half_profit_usd: float
    auto_half_fraction: float
    half_close_cooldown_bars: int
    warmup_risk_days: int
    warmup_risk_multiplier: float
    primary_tp_reward_multiple: float
    cooldown_bars_after_trade: int
    startup_warmup_bars: int
    max_hold_minutes: int
    loop_seconds: int
    lookback_bars: int
    htf_lookback_bars: int
    max_concentration_share: float
    min_positive_days_for_concentration: int
    allow_pyramiding: bool
    allow_foreign_positions: bool
    state_path: str
    log_file: str
    terminal_path: Optional[str]
    deviation: int
    magic: int
    log_level: str
    profit_close_usd: float = 0.0
    profit_close_pause_minutes: int = 0
    loss_close_pause_minutes: int = 0
    order_comment: str = ""
    chop_gate: str = "none"
    chop_adx_max: float = 18.0
    chop_efficiency_max: float = 0.18
    chop_atr_ratio_max: float = 0.85
    chop_slope_atr_max: float = 1.00
    chop_alternation_min: float = 0.55
    chop_min_score: float = 0.65
    chop_min_points: int = 3
    chop_non_asia_risk_mult: float = 0.25
    min_abs_score: float = 0.0
    min_adx: float = 0.0
    require_raw_htf_agree: bool = False
    entry_mode: str = "immediate"
    pullback_max_atr: float = 0.35
    enable_htf_lag_reversal_guard: bool = False
    htf_lag_momentum_threshold: float = 0.70
    htf_lag_m15_threshold: float = 0.50
    htf_lag_close_sma_buffer_atr: float = 0.05
    signal_reversal_take_profit_bars: int = 0
    signal_reversal_take_profit_window: int = 0
    signal_reversal_take_profit_count: int = 0
    signal_reversal_profit_only: bool = False
    signal_reversal_pause_minutes: int = 0
    force_close_friday_hour_utc: int = -1
    force_close_friday_minute_utc: int = 0
    vol_lookback: int = 48
    vol_ratio_weight: float = 0.15
    accel_weight: float = 0.10
    enable_dynamic_tp: bool = False
    enable_proactive_ipc: bool = False
    proactive_ipc_interval: int = 30


@dataclasses.dataclass
class MarketSnapshot:
    bar_time: dt.datetime
    close: float
    high: float
    low: float
    atr: float
    fast_sma: float
    slow_sma: float
    htf_fast_sma: float
    htf_slow_sma: float
    htf_signal: str
    compensated_htf_signal: str
    spread_points: float
    momentum: float
    m15_momentum: float
    score: float
    signal: str
    signal_source: str
    session: str
    chop_is_chop: bool
    chop_points: int
    chop_reason: str
    chop_risk_multiplier: float
    false_breakout_signal: str
    false_breakout_reason: str
    vol_ratio: float = 1.0
    accel_score: float = 0.0


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
    last_day_profit: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    loss_cooldown_until: Optional[str] = None
    loss_cooldown_triggered_at: Optional[str] = None
    profit_pause_until: Optional[str] = None
    profit_pause_triggered_at: Optional[str] = None
    loss_pause_until: Optional[str] = None
    loss_pause_triggered_at: Optional[str] = None
    auto_half_close_done: bool = False
    last_trade_bar_time: Optional[str] = None
    last_half_close_bar_time: Optional[str] = None
    trail_profit_since_by_ticket: Dict[str, str] = dataclasses.field(default_factory=dict)
    trail_cooldown_until_by_direction: Dict[str, str] = dataclasses.field(default_factory=dict)
    trailing_sl_by_ticket: Dict[str, float] = dataclasses.field(default_factory=dict)
    trailing_direction_by_ticket: Dict[str, str] = dataclasses.field(default_factory=dict)
    risk_warmup_started_at: Optional[str] = None
    last_processed_deal_time: Optional[str] = None
    last_processed_deal_ticket: int = 0
    last_close_profit: float = 0.0
    signal_reversal_history: List[str] = dataclasses.field(default_factory=list)
    dd_cooldown_until: Optional[str] = None
    dd_cooldown_triggered_at: Optional[str] = None
    paused_reason: str = ""
    paused: bool = False


class XAUUSDTrendStrategy:
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.mt5 = MetaTrader5(host=config.host, port=config.port)
        self.symbol = config.symbol
        self.timeframe = self._resolve_timeframe(config.timeframe)
        self.state_path = Path(config.state_path)
        self.log_file = Path(config.log_file)
        self.state = self._load_state()
        self._last_bar_time: Optional[dt.datetime] = None
        self._equity_fail_streak = 0
        self._connect_fail_streak = 0
        self._rates_fail_streak = 0
        self._last_reconnect_ts = 0.0
        self._last_signal = "NONE"
        self._startup_bars_seen = 0
        self._initialized = False
        self._symbol_point: Optional[float] = None
        self._symbol_digits: Optional[int] = None
        self._symbol_volume_min: float = 0.01
        self._symbol_volume_max: float = 100.0
        self._symbol_volume_step: float = 0.01
        self._symbol_contract_size: float = 1.0
        self._recent_primary_signals: Deque[str] = deque(maxlen=48)
        self._deal_sync_failed: bool = False
        self._prev_owned_positions: int = 0
        self._last_score = 0.0
        self._proactive_ipc_counter = 0

    def run(self) -> None:
        self._connect()
        self._prepare_symbol()

        while True:
            try:
                snapshot = self._build_snapshot()
                if not self._initialized:
                    self._ensure_day_context(snapshot.bar_time)
                    self._seed_equity()
                    self._log("Strategy started: %s %s live=%s", self.symbol, self.config.timeframe, self.config.live)
                    self._initialized = True

                self._ensure_day_context(snapshot.bar_time)
                self._maybe_force_close_friday(snapshot.bar_time)
                self._maybe_profit_close_on_loop()
                self._maybe_auto_half_close_on_loop()
                self._maybe_trail_on_loop()
                self._risk_guard(snapshot)

                if self._last_bar_time is not None and snapshot.bar_time <= self._last_bar_time:
                    time.sleep(self.config.loop_seconds)
                    continue

                self._last_bar_time = snapshot.bar_time
                self._sync_closed_trades(snapshot.bar_time)
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
                positive_days_profit=float(data.get("positive_days_profit", 0.0)),
                positive_days_count=int(data.get("positive_days_count", 0)),
                best_day_profit=float(data.get("best_day_profit", 0.0)),
                last_day_profit=float(data.get("last_day_profit", 0.0)),
                trades_today=int(data.get("trades_today", 0)),
                consecutive_losses=int(data.get("consecutive_losses", 0)),
                loss_cooldown_until=data.get("loss_cooldown_until"),
                loss_cooldown_triggered_at=data.get("loss_cooldown_triggered_at"),
                profit_pause_until=data.get("profit_pause_until"),
                profit_pause_triggered_at=data.get("profit_pause_triggered_at"),
                loss_pause_until=data.get("loss_pause_until"),
                loss_pause_triggered_at=data.get("loss_pause_triggered_at"),
                auto_half_close_done=bool(data.get("auto_half_close_done", False)),
                last_trade_bar_time=data.get("last_trade_bar_time"),
                last_half_close_bar_time=data.get("last_half_close_bar_time"),
                trail_profit_since_by_ticket={
                    str(k): str(v)
                    for k, v in dict(data.get("trail_profit_since_by_ticket", {})).items()
                },
                trail_cooldown_until_by_direction={
                    str(k).upper(): str(v)
                    for k, v in dict(data.get("trail_cooldown_until_by_direction", {})).items()
                },
                trailing_sl_by_ticket={
                    str(k): float(v)
                    for k, v in dict(data.get("trailing_sl_by_ticket", {})).items()
                },
                trailing_direction_by_ticket={
                    str(k): str(v).upper()
                    for k, v in dict(data.get("trailing_direction_by_ticket", {})).items()
                },
                risk_warmup_started_at=data.get("risk_warmup_started_at"),
                last_processed_deal_time=data.get("last_processed_deal_time"),
                last_processed_deal_ticket=int(data.get("last_processed_deal_ticket", 0) or 0),
                last_close_profit=float(data.get("last_close_profit", 0.0)),
                signal_reversal_history=[
                    str(item).upper()
                    for item in list(data.get("signal_reversal_history", []))[-12:]
                ],
                dd_cooldown_until=data.get("dd_cooldown_until"),
                dd_cooldown_triggered_at=data.get("dd_cooldown_triggered_at"),
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
            "positive_days_profit": self.state.positive_days_profit,
            "positive_days_count": self.state.positive_days_count,
            "best_day_profit": self.state.best_day_profit,
            "last_day_profit": self.state.last_day_profit,
            "trades_today": self.state.trades_today,
            "consecutive_losses": self.state.consecutive_losses,
            "loss_cooldown_until": self.state.loss_cooldown_until,
            "loss_cooldown_triggered_at": self.state.loss_cooldown_triggered_at,
            "profit_pause_until": self.state.profit_pause_until,
            "profit_pause_triggered_at": self.state.profit_pause_triggered_at,
            "loss_pause_until": self.state.loss_pause_until,
            "loss_pause_triggered_at": self.state.loss_pause_triggered_at,
            "auto_half_close_done": self.state.auto_half_close_done,
            "last_trade_bar_time": self.state.last_trade_bar_time,
            "last_half_close_bar_time": self.state.last_half_close_bar_time,
            "trail_profit_since_by_ticket": self.state.trail_profit_since_by_ticket,
            "trail_cooldown_until_by_direction": self.state.trail_cooldown_until_by_direction,
            "trailing_sl_by_ticket": self.state.trailing_sl_by_ticket,
            "trailing_direction_by_ticket": self.state.trailing_direction_by_ticket,
            "risk_warmup_started_at": self.state.risk_warmup_started_at,
            "last_processed_deal_time": self.state.last_processed_deal_time,
            "last_processed_deal_ticket": self.state.last_processed_deal_ticket,
            "last_close_profit": self.state.last_close_profit,
            "signal_reversal_history": self.state.signal_reversal_history,
            "dd_cooldown_until": self.state.dd_cooldown_until,
            "dd_cooldown_triggered_at": self.state.dd_cooldown_triggered_at,
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

    def _chart_now(self) -> dt.datetime:
        """MT5 tick time → chart/EET naive; fallback to UTC naive for tests."""
        try:
            tick = self.mt5.symbol_info_tick(self.config.symbol)
            tick_time = int(getattr(tick, "time", 0) or 0) if tick is not None else 0
            if tick_time > 0:
                return self._mt5_timestamp_to_chart_time(tick_time)
        except Exception:
            pass
        return dt.datetime.now(dt.UTC).replace(tzinfo=None)

    def _history_query_now(self) -> dt.datetime:
        """Local naive datetime for bridge history_deals_get query window."""
        return dt.datetime.now()

    def _connect(self) -> None:
        if self.config.terminal_path:
            self._log("Initializing terminal path: %s", self.config.terminal_path)
            ok = self.mt5.initialize(path=self.config.terminal_path)
        else:
            self._log("Initializing MT5 bridge without explicit terminal path")
            ok = self.mt5.initialize()
        self._log("initialize() -> %s | last_error=%s", ok, self.mt5.last_error())
        if not self._is_connected():
            raise RuntimeError(f"MT5 bridge not ready: {self.mt5.last_error()}")

    def _is_connected(self) -> bool:
        try:
            ti = self.mt5.terminal_info()
            ai = self.mt5.account_info()
            return ti is not None and ai is not None
        except Exception:
            return False

    def _is_transient_ipc_issue(self, exc: Exception) -> bool:
        text = str(exc)
        return (
            "No IPC connection" in text
            or "IPC timeout" in text
            or "stream has been closed" in text
            or "copy_rates_from_pos returned None" in text
            or "copy_rates_from_pos unavailable" in text
            or "account_info unavailable" in text
        )

    def _handle_transient_ipc(self, exc: Exception) -> None:
        self._log("Transient IPC issue: %s", exc)
        now = time.time()
        if now - self._last_reconnect_ts < 10:
            time.sleep(10)
        else:
            time.sleep(2)
        self._last_reconnect_ts = time.time()
        try:
            self._shutdown()
        except Exception:
            pass
        try:
            self.mt5 = MetaTrader5(host=self.config.host, port=self.config.port)
            self._connect()
            self._prepare_symbol()
            # Give the terminal a short settle period so bar history is available again
            # before the next market-data request.
            time.sleep(max(self.config.loop_seconds, 5))
        except Exception as reconnect_exc:
            self._log("Reconnect after transient IPC failed: %s", reconnect_exc)
            time.sleep(max(self.config.loop_seconds, 15))

    def _shutdown(self) -> None:
        try:
            self.mt5.shutdown()
        except Exception:
            pass

    def _prepare_symbol(self) -> None:
        last_error = None
        info = None
        for attempt in range(3):
            try:
                if not self.mt5.symbol_select(self.symbol, True):
                    last_error = self.mt5.last_error()
                    self._log(
                        "symbol_select failed for %s (attempt %s/3): %s",
                        self.symbol,
                        attempt + 1,
                        last_error,
                    )
                else:
                    info = self.mt5.symbol_info(self.symbol)
                    if info is not None:
                        break
                    last_error = self.mt5.last_error()
                    self._log(
                        "symbol_info unavailable for %s (attempt %s/3): %s",
                        self.symbol,
                        attempt + 1,
                        last_error,
                    )
            except Exception as exc:
                last_error = exc
                self._log(
                    "symbol preparation exception for %s (attempt %s/3): %s",
                    self.symbol,
                    attempt + 1,
                    exc,
                )
            if attempt < 2:
                time.sleep(1)
                try:
                    self._shutdown()
                except Exception:
                    pass
                self._connect()
        if info is None:
            raise RuntimeError(f"symbol_info is None for {self.symbol}: {last_error}")
        self._cache_symbol_info(info)
        self._log(
            "Symbol ready: %s | digits=%s point=%s volume_min=%s volume_step=%s volume_max=%s",
            self.symbol,
            getattr(info, "digits", "?"),
            getattr(info, "point", "?"),
            getattr(info, "volume_min", "?"),
            getattr(info, "volume_step", "?"),
            getattr(info, "volume_max", "?"),
        )

    def _cache_symbol_info(self, info: Any) -> None:
        self._symbol_point = float(getattr(info, "point", 0.0) or self._symbol_point or 0.01)
        self._symbol_digits = int(getattr(info, "digits", 0) or self._symbol_digits or 2)
        self._symbol_volume_min = float(getattr(info, "volume_min", self._symbol_volume_min) or self._symbol_volume_min)
        self._symbol_volume_max = float(getattr(info, "volume_max", self._symbol_volume_max) or self._symbol_volume_max)
        self._symbol_volume_step = float(getattr(info, "volume_step", self._symbol_volume_step) or self._symbol_volume_step)
        self._symbol_contract_size = float(getattr(info, "trade_contract_size", self._symbol_contract_size) or self._symbol_contract_size)

    def _symbol_info_once(self) -> Tuple[Optional[Any], Any]:
        client = MetaTrader5(host=self.config.host, port=self.config.port)
        try:
            ok = client.initialize()
            if not ok:
                return None, client.last_error()
            try:
                client.symbol_select(self.symbol, True)
            except Exception:
                pass
            info = client.symbol_info(self.symbol)
            if info is not None:
                self._cache_symbol_info(info)
            return info, client.last_error()
        except Exception as exc:
            return None, exc
        finally:
            try:
                client.shutdown()
            except Exception:
                pass

    def _order_send_once(self, request: Dict[str, Any]) -> Tuple[Optional[Any], Any]:
        client = MetaTrader5(host=self.config.host, port=self.config.port)
        try:
            ok = client.initialize()
            if not ok:
                return None, client.last_error()
            try:
                client.symbol_select(self.symbol, True)
            except Exception:
                pass
            result = client.order_send(request)
            return result, client.last_error()
        except Exception as exc:
            return None, exc
        finally:
            try:
                client.shutdown()
            except Exception:
                pass

    def _account_info_once(self) -> Tuple[Optional[Any], Any]:
        client = MetaTrader5(host=self.config.host, port=self.config.port)
        try:
            ok = client.initialize()
            if not ok:
                return None, client.last_error()
            info = client.account_info()
            return info, client.last_error()
        except Exception as exc:
            return None, exc
        finally:
            try:
                client.shutdown()
            except Exception:
                pass

    def _tick_once(self) -> Tuple[Optional[Any], Any]:
        client = MetaTrader5(host=self.config.host, port=self.config.port)
        try:
            ok = client.initialize()
            if not ok:
                return None, client.last_error()
            try:
                client.symbol_select(self.symbol, True)
            except Exception:
                pass
            tick = client.symbol_info_tick(self.symbol)
            return tick, client.last_error()
        except Exception as exc:
            return None, exc
        finally:
            try:
                client.shutdown()
            except Exception:
                pass

    def _copy_rates_once(self, timeframe: int, bars: int) -> Tuple[Optional[Any], Any]:
        """Fetch rates through a short-lived pymt5linux client.

        On this Wine/MT5 bridge, long-lived pymt5linux clients can keep
        account_info()/symbol_info working while repeated copy_rates_from_pos()
        calls degrade into (-10004, 'No IPC connection').  Market-data pulls are
        isolated in a fresh client so each bar request avoids the stale-client
        path while the main client remains available for trading/order calls.
        """
        client = MetaTrader5(host=self.config.host, port=self.config.port)
        try:
            ok = client.initialize()
            if not ok:
                return None, client.last_error()
            try:
                client.symbol_select(self.symbol, True)
            except Exception:
                pass
            rates = client.copy_rates_from_pos(self.symbol, timeframe, 0, bars)
            return rates, client.last_error()
        except Exception as exc:
            return None, exc
        finally:
            try:
                client.shutdown()
            except Exception:
                pass

    def _seed_equity(self) -> None:
        equity = self._get_equity()
        if self.state.initial_equity is None:
            self.state.initial_equity = equity
        if self.state.max_equity_seen is None:
            self.state.max_equity_seen = equity
        if self.state.day_start_equity is None:
            self.state.day_start_equity = equity
        if self.state.risk_warmup_started_at is None and self.config.warmup_risk_days > 0:
            self.state.risk_warmup_started_at = dt.datetime.now().isoformat(timespec="seconds")
        self.state.last_equity = equity
        self._save_state()
        self._log("Equity seeded: %.2f", equity)

    def _get_equity(self) -> float:
        for attempt in range(3):
            info, error = self._account_info_once()
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
                error,
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
        if self.state.current_day is None:
            self.state.current_day = day_key
            self.state.day_start_equity = self._get_equity()
            self.state.last_equity = self.state.day_start_equity
            self._save_state()
            return

        if self.state.current_day == day_key:
            return

        self._finalize_previous_day()
        self.state.current_day = day_key
        current_equity = self._get_equity()
        self.state.day_start_equity = current_equity
        self.state.last_equity = current_equity
        # Daily reset: allow the next trading day to start fresh.
        # Keep equity baselines, but clear counters that should not carry across days.
        # Also advance the closed-trade cursor so the next sync does not replay
        # historical deals and immediately restore stale loss counts.
        self.state.trades_today = 0
        self.state.consecutive_losses = 0
        self.state.loss_cooldown_until = None
        self.state.loss_cooldown_triggered_at = None
        self.state.auto_half_close_done = False
        self.state.last_processed_deal_time = self._chart_now().isoformat(timespec="seconds")
        self.state.last_processed_deal_ticket = 0
        self.state.paused = False
        self.state.paused_reason = ""
        self._save_state()
        self._log(
            "New trading day: %s | day_start_equity=%.2f | trades_today=%d | consecutive_losses=%d | last_processed_deal_time=%s",
            day_key,
            current_equity,
            self.state.trades_today,
            self.state.consecutive_losses,
            self.state.last_processed_deal_time,
        )

    def _finalize_previous_day(self) -> None:
        if self.state.day_start_equity is None or self.state.last_equity is None:
            return
        day_profit = self.state.last_equity - self.state.day_start_equity
        self.state.last_day_profit = day_profit
        if day_profit > 0:
            self.state.positive_days_profit += day_profit
            self.state.positive_days_count += 1
            if day_profit > self.state.best_day_profit:
                self.state.best_day_profit = day_profit
        self._log(
            "Day finalized | profit=%.2f | positive_days_profit=%.2f | positive_days_count=%d | best_day=%.2f | best_share=%.2f%%",
            day_profit,
            self.state.positive_days_profit,
            self.state.positive_days_count,
            self.state.best_day_profit,
            self._best_day_share() * 100.0,
        )
        self._save_state()

    def _best_day_share(self) -> float:
        if self.state.positive_days_profit <= 0:
            return 0.0
        return float(self.state.best_day_profit) / float(self.state.positive_days_profit)

    def _maybe_resume_after_dd_cooldown(self, equity: float) -> None:
        if not self.state.paused or not self.state.dd_cooldown_until:
            return
        if not self.state.paused_reason.startswith(("Daily drawdown", "Total drawdown")):
            return
        try:
            until = dt.datetime.fromisoformat(str(self.state.dd_cooldown_until))
        except ValueError:
            self._log("Invalid DD cooldown timestamp: %s", self.state.dd_cooldown_until)
            return
        now = self._chart_now()
        if now < until:
            self._log("DD cooldown active: until=%s reason=%s", self.state.dd_cooldown_until, self.state.paused_reason)
            return
        self.state.day_start_equity = equity
        self.state.initial_equity = equity
        self.state.max_equity_seen = equity
        self.state.last_equity = equity
        self.state.trades_today = 0
        self.state.consecutive_losses = 0
        self.state.loss_cooldown_until = None
        self.state.loss_cooldown_triggered_at = None
        self.state.profit_pause_until = None
        self.state.profit_pause_triggered_at = None
        self.state.dd_cooldown_until = None
        self.state.dd_cooldown_triggered_at = None
        self.state.paused = False
        self.state.paused_reason = ""
        self.state.last_processed_deal_time = now.isoformat(timespec="seconds")
        self.state.last_processed_deal_ticket = 0
        self._log("DD cooldown expired; resumed with fresh baseline equity=%.2f", equity)
        self._save_state()

    def _risk_guard(self, snapshot: MarketSnapshot) -> None:
        equity = self._get_equity()
        self.state.last_equity = equity
        if self.state.initial_equity is None:
            self.state.initial_equity = equity
        if self.state.day_start_equity is None:
            self.state.day_start_equity = equity
        if self.state.max_equity_seen is None or equity > self.state.max_equity_seen:
            self.state.max_equity_seen = equity

        day_start = max(float(self.state.day_start_equity), 1e-9)
        max_equity = max(float(self.state.max_equity_seen), 1e-9)
        initial_equity = max(float(self.state.initial_equity), 1e-9)

        daily_dd = 1.0 - (equity / day_start)
        total_dd = 1.0 - (equity / max_equity)
        profit_progress = equity / initial_equity - 1.0

        self._maybe_resume_after_dd_cooldown(equity)

        self._log(
            "Risk | equity=%.2f day_dd=%.2f%% total_dd=%.2f%% profit=%.2f%% best_share=%.2f%% paused=%s",
            equity,
            daily_dd * 100.0,
            total_dd * 100.0,
            profit_progress * 100.0,
            self._best_day_share() * 100.0,
            self.state.paused,
        )
        self._save_state()

        if daily_dd >= self.config.daily_dd_limit:
            now = self._chart_now()
            until = now + dt.timedelta(hours=4)
            self.state.paused = True
            self.state.paused_reason = f"Daily drawdown {daily_dd * 100.0:.2f}% >= {self.config.daily_dd_limit * 100.0:.2f}%"
            self.state.dd_cooldown_triggered_at = now.isoformat(timespec="seconds")
            self.state.dd_cooldown_until = until.isoformat(timespec="seconds")
            self._log("%s; DD cooldown until %s", self.state.paused_reason, self.state.dd_cooldown_until)
            self.close_all_positions()
            self._save_state()
            raise SystemExit(1)

        if total_dd >= self.config.total_dd_limit:
            now = self._chart_now()
            until = now + dt.timedelta(hours=4)
            self.state.paused = True
            self.state.paused_reason = f"Total drawdown {total_dd * 100.0:.2f}% >= {self.config.total_dd_limit * 100.0:.2f}%"
            self.state.dd_cooldown_triggered_at = now.isoformat(timespec="seconds")
            self.state.dd_cooldown_until = until.isoformat(timespec="seconds")
            self._log("%s; DD cooldown until %s", self.state.paused_reason, self.state.dd_cooldown_until)
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

        if self._concentration_guard(equity):
            self.state.paused = True
            self.state.paused_reason = "Best Day concentration guard active"
            self._log(self.state.paused_reason)
        elif self.state.paused and self.state.paused_reason == "Best Day concentration guard active":
            self.state.paused = False
            self.state.paused_reason = ""

    def _concentration_guard(self, equity: float) -> bool:
        if self.state.day_start_equity is None:
            return False
        if self.state.positive_days_count < self.config.min_positive_days_for_concentration:
            return False

        today_profit = max(0.0, equity - float(self.state.day_start_equity))
        prior_positive_profit = max(0.0, self.state.positive_days_profit)
        if today_profit <= 0:
            return False

        today_cap = max(
            float(self.state.initial_equity or self.config.start_equity) * 0.008,
            prior_positive_profit * self.config.max_concentration_share,
        )
        if today_profit >= today_cap:
            projected_total = prior_positive_profit + today_profit
            projected_best = max(self.state.best_day_profit, today_profit)
            projected_share = projected_best / projected_total if projected_total > 0 else 0.0
            self._log(
                "Concentration guard | today_profit=%.2f cap=%.2f projected_share=%.2f%%",
                today_profit,
                today_cap,
                projected_share * 100.0,
            )
            return True
        return False

    def _build_htf_filter(self) -> Tuple[float, float, str]:
        htf_timeframe_name = str(self.config.htf_timeframe).strip().upper() or "H1"
        htf_timeframe = self._resolve_timeframe(htf_timeframe_name)
        bars, error = self._copy_rates_once(htf_timeframe, self.config.htf_lookback_bars)
        if bars is None:
            raise RuntimeError(f"HTF copy_rates_from_pos failed: {error}")
        htf_bars = [self._normalize_bar(row) for row in list(bars)]
        if len(htf_bars) < max(self.config.htf_slow_sma, self.config.htf_fast_sma) + 3:
            raise RuntimeError(f"Not enough HTF bars: {len(htf_bars)}")
        closes = [bar["close"] for bar in htf_bars[:-1]]
        last_close = closes[-1]
        fast = statistics.fmean(closes[-self.config.htf_fast_sma:])
        slow = statistics.fmean(closes[-self.config.htf_slow_sma:])
        if len(closes) >= 4:
            recent_slope = closes[-1] - closes[-4]
        else:
            recent_slope = 0.0
        # Price-position override: when price is below both SMAs, the
        # SMA-cross signal is lagging and misleading — force BEAR.
        # Likewise when price is above both SMAs, force BULL.
        if last_close < fast and last_close < slow:
            return fast, slow, "BEAR"
        if last_close > fast and last_close > slow:
            return fast, slow, "BULL"
        # Fallback to SMA cross with slope confirmation.
        if fast > slow and recent_slope >= 0:
            return fast, slow, "BULL"
        if fast < slow and recent_slope <= 0:
            return fast, slow, "BEAR"
        if fast > slow:
            return fast, slow, "BULL"
        if fast < slow:
            return fast, slow, "BEAR"
        return fast, slow, "NEUTRAL"

    def _m15_momentum(self, atr_ref: float) -> float:
        m15_timeframe = self._resolve_timeframe("M15")
        bars, error = self._copy_rates_once(m15_timeframe, max(20, self.config.breakout_lookback + 5))
        if bars is None:
            raise RuntimeError(f"M15 copy_rates_from_pos failed: {error}")
        m15_bars = [self._normalize_bar(row) for row in list(bars)]
        if len(m15_bars) < 5 or atr_ref <= 0:
            return 0.0
        closes = [bar["close"] for bar in m15_bars[:-1]]
        if len(closes) < 4:
            return 0.0
        momentum = (closes[-1] - closes[-4]) / atr_ref
        return self._clamp(momentum, -2.0, 2.0)

    def _compensated_htf_signal(self, htf_signal: str, m15_momentum: float) -> str:
        """Use strong M15 momentum to compensate for a lagging H1 filter."""
        compensation_threshold = max(
            float(self.config.htf_comp_momentum_threshold),
            float(self.config.trend_threshold) * 2.0,
        )
        if m15_momentum >= compensation_threshold:
            return "BULL"
        if m15_momentum <= -compensation_threshold:
            return "BEAR"
        return htf_signal

    def _spread_points(self) -> float:
        tick, error = self._tick_once()
        if tick is None:
            raise RuntimeError(f"symbol_info_tick failed for spread: {error}")
        ask = float(getattr(tick, "ask", 0.0) or 0.0)
        bid = float(getattr(tick, "bid", 0.0) or 0.0)
        point = self._point()
        if ask <= 0 or bid <= 0 or point <= 0:
            return 0.0
        return abs(ask - bid) / point

    def _session_allowed(self, bar_time: dt.datetime) -> bool:
        hour = bar_time.hour
        start = int(self.config.session_start_utc)
        end = int(self.config.session_end_utc)
        if start == end:
            return True
        if start < end:
            return start <= hour < end
        return hour >= start or hour < end

    def _cooldown_ok(self, bar_time: dt.datetime) -> bool:
        if self._half_close_cooldown_active(bar_time):
            return False
        if self.state.last_trade_bar_time is None:
            return True
        try:
            last_dt = dt.datetime.fromisoformat(str(self.state.last_trade_bar_time))
        except Exception:
            return True
        delta_bars = (bar_time - last_dt).total_seconds() / self._timeframe_seconds()
        return delta_bars >= float(self.config.cooldown_bars_after_trade)

    def _half_close_cooldown_active(self, bar_time: dt.datetime) -> bool:
        cooldown_bars = max(0, int(self.config.half_close_cooldown_bars))
        if cooldown_bars <= 0 or not self.state.last_half_close_bar_time:
            return False
        try:
            last_dt = dt.datetime.fromisoformat(str(self.state.last_half_close_bar_time))
        except ValueError:
            self.state.last_half_close_bar_time = None
            self._save_state()
            return False
        delta_bars = (bar_time - last_dt).total_seconds() / self._timeframe_seconds()
        if delta_bars < float(cooldown_bars):
            self._log(
                "Half-close cooldown blocks new entry: %.1f/%d bars since %s",
                delta_bars,
                cooldown_bars,
                self.state.last_half_close_bar_time,
            )
            return True
        return False

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

    def _loss_cooldown_active(self, now: Optional[dt.datetime] = None) -> bool:
        if not self.state.loss_cooldown_until:
            return False
        try:
            until = dt.datetime.fromisoformat(str(self.state.loss_cooldown_until))
        except ValueError:
            self.state.loss_cooldown_until = None
            self.state.loss_cooldown_triggered_at = None
            self._save_state()
            return False
        current = now or self._chart_now()
        if current < until:
            return True
        self._log("Loss cooldown expired: until=%s", self.state.loss_cooldown_until)
        self.state.loss_cooldown_until = None
        self.state.loss_cooldown_triggered_at = None
        self.state.consecutive_losses = 0
        self._log("Consecutive losses reset to 0 after cooldown expiry")
        self._save_state()
        return False

    def _profit_pause_active(self, now: Optional[dt.datetime] = None) -> bool:
        if not self.state.profit_pause_until:
            return False
        try:
            until = dt.datetime.fromisoformat(str(self.state.profit_pause_until))
        except ValueError:
            self.state.profit_pause_until = None
            self.state.profit_pause_triggered_at = None
            self._save_state()
            return False
        current = now or self._chart_now()
        if current < until:
            return True
        self._log("Profit-close pause expired: until=%s", self.state.profit_pause_until)
        self.state.profit_pause_until = None
        self.state.profit_pause_triggered_at = None
        self._save_state()
        return False

    def _activate_profit_pause(self, floating_profit: float) -> None:
        minutes = int(self.config.profit_close_pause_minutes)
        if minutes <= 0:
            return
        now = self._chart_now()
        until = now + dt.timedelta(minutes=minutes)
        self.state.profit_pause_triggered_at = now.isoformat(timespec="seconds")
        self.state.profit_pause_until = until.isoformat(timespec="seconds")
        self._log(
            "Profit-close pause active: floating_profit=%.2f until=%s",
            floating_profit,
            self.state.profit_pause_until,
        )
        self._save_state()

    def _loss_close_pause_active(self, now: Optional[dt.datetime] = None) -> bool:
        if not self.state.loss_pause_until:
            return False
        try:
            until = dt.datetime.fromisoformat(str(self.state.loss_pause_until))
        except ValueError:
            self.state.loss_pause_until = None
            self.state.loss_pause_triggered_at = None
            self._save_state()
            return False
        current = now or self._chart_now()
        if current < until:
            return True
        self._log("Loss-close pause expired: until=%s", self.state.loss_pause_until)
        self.state.loss_pause_until = None
        self.state.loss_pause_triggered_at = None
        self._save_state()
        return False

    def _activate_loss_close_pause(self, close_profit: float, closed_at: Optional[dt.datetime] = None) -> None:
        minutes = int(self.config.loss_close_pause_minutes)
        if minutes <= 0:
            return
        if self._loss_close_pause_active():
            return
        trigger_at = closed_at or self._chart_now()
        until = trigger_at + dt.timedelta(minutes=minutes)
        self.state.loss_pause_triggered_at = trigger_at.isoformat(timespec="seconds")
        self.state.loss_pause_until = until.isoformat(timespec="seconds")
        self._log(
            "Loss-close pause active: close_profit=%.2f closed_at=%s until=%s",
            close_profit,
            self.state.loss_pause_triggered_at,
            self.state.loss_pause_until,
        )

    def _activate_loss_cooldown(self, closed_at: Optional[dt.datetime] = None) -> None:
        threshold = int(self.config.loss_cooldown_losses)
        minutes = int(self.config.loss_cooldown_minutes)
        if threshold <= 0 or minutes <= 0:
            return
        if self.state.consecutive_losses < threshold:
            return
        if self._loss_cooldown_active():
            return
        trigger_at = closed_at or self._chart_now()
        until = trigger_at + dt.timedelta(minutes=minutes)
        self.state.loss_cooldown_triggered_at = trigger_at.isoformat(timespec="seconds")
        self.state.loss_cooldown_until = until.isoformat(timespec="seconds")
        self._log(
            "Loss cooldown active: consecutive_losses=%d threshold=%d closed_at=%s until=%s",
            self.state.consecutive_losses,
            threshold,
            self.state.loss_cooldown_triggered_at,
            self.state.loss_cooldown_until,
        )


    def _friday_force_close_cutoff_reached(self, bar_time: dt.datetime) -> bool:
        cutoff_hour = int(self.config.force_close_friday_hour_utc)
        if cutoff_hour < 0:
            return False
        cutoff_minute = int(self.config.force_close_friday_minute_utc)
        if not (0 <= cutoff_hour <= 23 and 0 <= cutoff_minute <= 59):
            return False
        if bar_time.weekday() != 4:
            return False
        cutoff = bar_time.replace(hour=cutoff_hour, minute=cutoff_minute, second=0, microsecond=0)
        return bar_time >= cutoff

    def _new_entry_block_reason(self) -> Optional[str]:
        if self.state.paused:
            return f"paused: {self.state.paused_reason}"
        if self._profit_pause_active():
            return f"profit-close pause until {self.state.profit_pause_until}"
        if self._loss_close_pause_active():
            return f"loss-close pause until {self.state.loss_pause_until}"
        if self.state.consecutive_losses >= int(self.config.max_consecutive_losses):
            if self._loss_cooldown_active():
                return (
                    f"loss cooldown until {self.state.loss_cooldown_until} "
                    f"after {self.state.consecutive_losses} consecutive losses"
                )
            self._log(
                "Consecutive loss cap (%d) reached but cooldown expired; resetting",
                self.state.consecutive_losses,
            )
            self.state.consecutive_losses = 0
            self._save_state()
        elif self._loss_cooldown_active():
            return (
                f"loss cooldown until {self.state.loss_cooldown_until} "
                f"after {self.state.consecutive_losses} consecutive losses"
            )
        return None

    def _deal_net_profit(self, deal: Any) -> float:
        return (
            float(getattr(deal, "profit", 0.0) or 0.0)
            + float(getattr(deal, "commission", 0.0) or 0.0)
            + float(getattr(deal, "swap", 0.0) or 0.0)
        )

    def _is_owned_closing_deal(self, deal: Any) -> bool:
        symbol = str(getattr(deal, "symbol", "") or "")
        if symbol != self.config.symbol:
            return False
        magic = int(getattr(deal, "magic", 0) or 0)
        if magic != int(self.config.magic):
            return False
        entry = getattr(deal, "entry", None)
        close_entries = {
            getattr(self.mt5, "DEAL_ENTRY_OUT", 1),
            getattr(self.mt5, "DEAL_ENTRY_OUT_BY", 3),
        }
        return entry is None or int(entry) in close_entries

    def _mt5_timestamp_to_chart_time(self, timestamp: int) -> dt.datetime:
        """Convert an MT5 timestamp to the naive chart/server time shown in MT5.

        `pymt5linux` returns MT5 chart timestamps as epoch seconds. Using
        `fromtimestamp()` applies the Linux process timezone and shifts the
        displayed chart hour, which breaks session gates. Converting from UTC
        and dropping tzinfo preserves the MT5 chart time basis used by the
        terminal display while keeping the strategy's naive-datetime state.
        """
        return dt.datetime.fromtimestamp(int(timestamp), dt.UTC).replace(tzinfo=None)

    def _max_hold_exceeded(self, pos: PositionState, bar_time: dt.datetime) -> bool:
        if int(self.config.max_hold_minutes) <= 0:
            return False
        if pos.time_open is None:
            return False
        opened = self._mt5_timestamp_to_chart_time(int(pos.time_open))
        held_minutes = (bar_time - opened).total_seconds() / 60.0
        return held_minutes >= float(self.config.max_hold_minutes)

    def _direction_from_position_type(self, position_type: int) -> str:
        return "BUY" if position_type == self.mt5.POSITION_TYPE_BUY else "SELL"

    def _opposite_direction(self, direction: str) -> str:
        return "SELL" if direction == "BUY" else "BUY"

    def _direction_cooldown_active(self, direction: str, now: Optional[dt.datetime] = None) -> bool:
        direction_key = direction.upper()
        raw_until = self.state.trail_cooldown_until_by_direction.get(direction_key)
        if not raw_until:
            return False
        try:
            until = dt.datetime.fromisoformat(str(raw_until))
        except ValueError:
            self.state.trail_cooldown_until_by_direction.pop(direction_key, None)
            self._save_state()
            return False
        current = now or self._chart_now()
        if current < until:
            self._log(
                "Trailing-stop cooldown blocks %s entry until %s",
                direction_key,
                until.isoformat(timespec="seconds"),
            )
            return True
        self.state.trail_cooldown_until_by_direction.pop(direction_key, None)
        self._save_state()
        return False

    def _activate_trailing_stop_cooldown(self, closed_direction: str, deal_iso: str) -> None:
        minutes = int(self.config.trail_same_direction_cooldown_minutes)
        if minutes <= 0:
            return
        direction_key = closed_direction.upper()
        closed_at = dt.datetime.fromisoformat(deal_iso)
        until = closed_at + dt.timedelta(minutes=minutes)
        prev_raw = self.state.trail_cooldown_until_by_direction.get(direction_key)
        if prev_raw:
            try:
                prev = dt.datetime.fromisoformat(prev_raw)
                until = max(until, prev)
            except ValueError:
                pass
        self.state.trail_cooldown_until_by_direction[direction_key] = until.isoformat(timespec="seconds")
        self._log(
            "Trailing-stop cooldown active: direction=%s until=%s",
            direction_key,
            self.state.trail_cooldown_until_by_direction[direction_key],
        )

    def _sync_trailing_stop_cooldowns(self, deal: Any, deal_iso: str) -> None:
        if int(getattr(deal, "entry", -1)) != 1:
            return
        position_id = str(getattr(deal, "position_id", "") or getattr(deal, "order", "") or "")
        if not position_id:
            return
        tracked_sl = self.state.trailing_sl_by_ticket.pop(position_id, None)
        closed_direction = self.state.trailing_direction_by_ticket.pop(position_id, "")
        self.state.trail_profit_since_by_ticket.pop(position_id, None)
        if tracked_sl is None:
            return
        exit_price = float(getattr(deal, "price", 0.0) or 0.0)
        if exit_price <= 0:
            return
        tolerance = max(self._point() * 50.0, 0.5)
        if abs(exit_price - float(tracked_sl)) > tolerance:
            return
        if closed_direction not in {"BUY", "SELL"}:
            deal_type = int(getattr(deal, "type", -1))
            if deal_type == self.mt5.DEAL_TYPE_SELL:
                closed_direction = "BUY"
            elif deal_type == self.mt5.DEAL_TYPE_BUY:
                closed_direction = "SELL"
        if closed_direction not in {"BUY", "SELL"}:
            return
        self._activate_trailing_stop_cooldown(closed_direction, deal_iso)

    def _sync_closed_trades(self, bar_time: Optional[dt.datetime] = None) -> None:
        reference_bar_time = bar_time or self._last_bar_time or self._chart_now()
        to_time = self._history_query_now()
        # The Wine/pymt5linux bridge can miss late same-day deals when the query
        # end is before the broker/chart day boundary. Query through the next
        # chart-day buffer, then filter by the persistent deal cursor below.
        history_end_time = max(
            to_time,
            reference_bar_time.replace(hour=0, minute=0, second=0, microsecond=0)
            + dt.timedelta(days=1, hours=3),
        )
        # Use the chart/server day as lower bound and rely on the persistent
        # deal cursor below to skip old deals. The Wine/pymt5linux bridge can
        # omit newer same-day deals when the lower bound is near the cursor.
        from_time = reference_bar_time.replace(hour=0, minute=0, second=0, microsecond=0)
        deals = None
        for attempt in range(3):
            try:
                deals = self.mt5.history_deals_get(from_time, history_end_time)
                if deals is None:
                    # history_deals_get returns None on degraded IPC, not an
                    # exception. Reinitialize the bridge connection and retry.
                    self._log(
                        "history_deals_get returned None (attempt %d/3); reinitializing bridge",
                        attempt + 1,
                    )
                    if attempt < 2:
                        self.mt5.shutdown()
                        time.sleep(1)
                        self._connect()
                        time.sleep(1)
                    continue
                break
            except Exception as exc:
                self._log(
                    "Closed deal sync query failed (attempt %d/3): %s",
                    attempt + 1,
                    exc,
                )
                if attempt < 2:
                    time.sleep(2)
        if deals is None:
            self._log("Closed deal sync skipped: history_deals_get failed after 3 attempts")
            self._deal_sync_failed = True
            return
        self._deal_sync_failed = False
        if not deals:
            return
        latest_seen = self.state.last_processed_deal_time
        latest_seen_ticket = int(self.state.last_processed_deal_ticket or 0)
        ordered_deals = sorted(
            list(deals),
            key=lambda item: (int(getattr(item, "time", 0) or 0), int(getattr(item, "ticket", 0) or 0)),
        )
        for deal in ordered_deals:
            deal_time = getattr(deal, "time", None)
            if deal_time is None:
                continue
            deal_ticket = int(getattr(deal, "ticket", 0) or 0)
            deal_dt = self._mt5_timestamp_to_chart_time(int(deal_time))
            deal_iso = deal_dt.isoformat()
            if latest_seen is not None:
                if deal_iso < latest_seen:
                    continue
                if deal_iso == latest_seen and deal_ticket <= latest_seen_ticket:
                    continue
            if not self._is_owned_closing_deal(deal):
                continue
            closed_at = deal_dt
            profit = self._deal_net_profit(deal)
            self.state.last_close_profit = profit
            self.state.last_processed_deal_time = deal_iso
            self.state.last_processed_deal_ticket = deal_ticket
            latest_seen = deal_iso
            latest_seen_ticket = deal_ticket
            self._sync_trailing_stop_cooldowns(deal, deal_iso)
            if abs(profit) < REALIZED_PNL_NOISE_USD:
                self._log(
                    "Closed deal sync ignored noise | time=%s symbol=%s magic=%s profit=%.2f consecutive_losses=%d",
                    deal_iso,
                    getattr(deal, "symbol", ""),
                    getattr(deal, "magic", ""),
                    profit,
                    self.state.consecutive_losses,
                )
                continue
            if profit < -5.0:
                self.state.consecutive_losses += 1
                self._activate_loss_close_pause(profit, closed_at)
                self._activate_loss_cooldown(closed_at)
            elif profit > 0:
                self.state.consecutive_losses = 0
                self.state.loss_cooldown_until = None
                self.state.loss_cooldown_triggered_at = None
                self.state.loss_pause_until = None
                self.state.loss_pause_triggered_at = None
            self._log(
                "Closed deal sync | time=%s symbol=%s magic=%s profit=%.2f consecutive_losses=%d",
                deal_iso,
                getattr(deal, "symbol", ""),
                getattr(deal, "magic", ""),
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

    def _build_snapshot(self) -> MarketSnapshot:
        bars = self._fetch_bars()
        if len(bars) < max(self.config.slow_sma, self.config.atr_period, self.config.breakout_lookback) + 5:
            raise RuntimeError(f"Not enough bars for signal calculation: {len(bars)}")

        last_closed = bars[-2]
        closes = [bar["close"] for bar in bars[:-1]]
        highs = [bar["high"] for bar in bars[:-1]]
        lows = [bar["low"] for bar in bars[:-1]]

        atr = self._atr(bars[:-1], self.config.atr_period)
        fast_sma = statistics.fmean(closes[-self.config.fast_sma:])
        slow_sma = statistics.fmean(closes[-self.config.slow_sma:])
        momentum = (closes[-1] - closes[-4]) / atr if atr > 0 and len(closes) >= 4 else 0.0
        momentum = self._clamp(momentum, -2.0, 2.0)
        htf_fast_sma, htf_slow_sma, htf_signal = self._build_htf_filter()
        m15_momentum = self._m15_momentum(atr)
        compensated_htf_signal = self._compensated_htf_signal(htf_signal, m15_momentum)
        spread_points = self._spread_points()

        vol_lookback = max(10, int(self.config.vol_lookback))
        baseline_bars = bars[:-1][-vol_lookback:]
        if len(baseline_bars) >= 12:
            bl_period = max(2, min(vol_lookback // 2, len(baseline_bars) - 1))
            baseline_atr = self._atr(baseline_bars, bl_period)
            vol_ratio = atr / max(baseline_atr, self._point() * 5)
        else:
            vol_ratio = 1.0

        accel_score = 0.0
        if atr > 0 and len(closes) >= 7:
            mom_short = closes[-1] - closes[-3]
            mom_long = closes[-1] - closes[-6]
            accel_score = self._clamp(abs(mom_short - mom_long) / max(atr, self._point() * 5), 0.0, 1.0)

        score = self._score_signal(
            closes,
            highs,
            lows,
            atr,
            fast_sma,
            slow_sma,
            htf_fast_sma,
            htf_slow_sma,
            compensated_htf_signal,
            momentum,
            m15_momentum,
            spread_points,
            vol_ratio=vol_ratio,
            accel_score=accel_score,
        )
        self._last_score = score
        signal = self._decide_signal(score, compensated_htf_signal, spread_points)
        signal_source = "trend" if signal in {"BUY", "SELL"} else "none"
        if signal_source == "trend":
            adx = self._calculate_adx(bars[:-1][-120:], 14)
            if not self._quality_filter_allows(
                signal=signal,
                htf_signal=htf_signal,
                compensated_htf_signal=compensated_htf_signal,
                score=score,
                adx=adx,
            ):
                self._log(
                    "Quality filter blocks %s: score=%.2f adx=%.2f raw_htf=%s comp_htf=%s min_score=%.2f min_adx=%.2f raw_agree=%s",
                    signal,
                    score,
                    adx,
                    htf_signal,
                    compensated_htf_signal,
                    float(self.config.min_abs_score),
                    float(self.config.min_adx),
                    bool(self.config.require_raw_htf_agree),
                )
                signal = "NONE"
                signal_source = "none"
        if signal_source == "trend" and self._htf_lag_reversal_blocks(
            signal=signal,
            htf_signal=htf_signal,
            close=float(last_closed["close"]),
            fast_sma=fast_sma,
            slow_sma=slow_sma,
            momentum=momentum,
            m15_momentum=m15_momentum,
            atr=atr,
        ):
            self._log(
                "HTF lag reversal guard blocks %s: htf=%s close=%.2f fast=%.2f slow=%.2f momentum=%.2f m15_mom=%.2f",
                signal,
                htf_signal,
                float(last_closed["close"]),
                fast_sma,
                slow_sma,
                momentum,
                m15_momentum,
            )
            signal = "NONE"
            signal_source = "none"
        if signal_source == "trend" and not self._trend_pullback_allows(
            signal=signal,
            closed_bars=bars[:-1],
            atr=atr,
            fast_sma=fast_sma,
        ):
            self._log(
                "Pullback entry gate blocks %s: close=%.2f high=%.2f low=%.2f fast=%.2f atr=%.2f max_atr=%.2f",
                signal,
                float(last_closed["close"]),
                float(last_closed["high"]),
                float(last_closed["low"]),
                fast_sma,
                atr,
                float(self.config.pullback_max_atr),
            )
            signal = "NONE"
            signal_source = "none"
        session = self._session_label(self._bar_time(last_closed))
        chop_signals = list(self._recent_primary_signals) + [signal]
        chop_is_chop, chop_points, chop_reason = self._detect_chop(
            bars[:-1],
            score,
            atr,
            fast_sma,
            slow_sma,
            chop_signals,
        )
        chop_risk_multiplier = self._chop_risk_multiplier(chop_is_chop, session)
        if signal_source == "trend" and chop_risk_multiplier <= 0.0:
            signal = "NONE"
            signal_source = "none"
        false_breakout_signal, false_breakout_reason = self._false_breakout_reversal_signal(
            bars[:-1],
            atr,
            compensated_htf_signal,
            spread_points,
        )
        if signal == "NONE" and false_breakout_signal != "NONE":
            signal = false_breakout_signal
            signal_source = "complement"
            chop_risk_multiplier = 1.0

        return MarketSnapshot(
            bar_time=self._bar_time(last_closed),
            close=float(last_closed["close"]),
            high=float(last_closed["high"]),
            low=float(last_closed["low"]),
            atr=atr,
            fast_sma=fast_sma,
            slow_sma=slow_sma,
            htf_fast_sma=htf_fast_sma,
            htf_slow_sma=htf_slow_sma,
            htf_signal=htf_signal,
            compensated_htf_signal=compensated_htf_signal,
            spread_points=spread_points,
            momentum=momentum,
            m15_momentum=m15_momentum,
            score=score,
            signal=signal,
            signal_source=signal_source,
            session=session,
            chop_is_chop=chop_is_chop,
            chop_points=chop_points,
            chop_reason=chop_reason,
            chop_risk_multiplier=chop_risk_multiplier,
            false_breakout_signal=false_breakout_signal,
            false_breakout_reason=false_breakout_reason,
            vol_ratio=vol_ratio,
            accel_score=accel_score,
        )

    def _fetch_bars(self) -> List[Dict[str, float]]:
        if self.config.enable_proactive_ipc:
            self._proactive_ipc_counter += 1
            interval = max(1, int(self.config.proactive_ipc_interval))
            if self._proactive_ipc_counter >= interval:
                self._proactive_ipc_counter = 0
                try:
                    self.mt5.shutdown()
                except Exception:
                    pass
                time.sleep(0.5)
                self.mt5 = MetaTrader5(host=self.config.host, port=self.config.port)
                ok = self.mt5.initialize()
                if ok:
                    self._log("Proactive IPC refresh: MT5 reinitialized.")
                else:
                    self._log("Proactive IPC refresh: MT5 reinit returned %s", ok)
        while True:
            for attempt in range(3):
                rates, error = self._copy_rates_once(self.timeframe, self.config.lookback_bars)
                if rates is not None:
                    self._rates_fail_streak = 0
                    bars: List[Dict[str, float]] = []
                    for row in list(rates):
                        bars.append(self._normalize_bar(row))
                    return bars

                self._rates_fail_streak += 1
                self._log(
                    "copy_rates_from_pos unavailable (attempt %s/3, streak=%s): %s",
                    attempt + 1,
                    self._rates_fail_streak,
                    error,
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
        return self._mt5_timestamp_to_chart_time(int(bar["time"]))

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

    def _true_ranges(self, bars: Sequence[Dict[str, float]]) -> List[float]:
        out: List[float] = []
        for i in range(1, len(bars)):
            curr = bars[i]
            prev = bars[i - 1]
            out.append(
                max(
                    float(curr["high"]) - float(curr["low"]),
                    abs(float(curr["high"]) - float(prev["close"])),
                    abs(float(curr["low"]) - float(prev["close"])),
                )
            )
        return out

    def _calculate_adx(self, bars: Sequence[Dict[str, float]], period: int = 14) -> float:
        if len(bars) < period * 2 + 2:
            return 50.0
        plus_dm: List[float] = []
        minus_dm: List[float] = []
        trs: List[float] = []
        for i in range(1, len(bars)):
            up_move = float(bars[i]["high"]) - float(bars[i - 1]["high"])
            down_move = float(bars[i - 1]["low"]) - float(bars[i]["low"])
            plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
            minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
            curr = bars[i]
            prev = bars[i - 1]
            trs.append(
                max(
                    float(curr["high"]) - float(curr["low"]),
                    abs(float(curr["high"]) - float(prev["close"])),
                    abs(float(curr["low"]) - float(prev["close"])),
                )
            )
        dxs: List[float] = []
        for idx in range(period, len(trs) + 1):
            tr_sum = sum(trs[idx - period:idx])
            if tr_sum <= 0:
                continue
            plus_di = 100.0 * sum(plus_dm[idx - period:idx]) / tr_sum
            minus_di = 100.0 * sum(minus_dm[idx - period:idx]) / tr_sum
            denom = plus_di + minus_di
            if denom <= 0:
                continue
            dxs.append(100.0 * abs(plus_di - minus_di) / denom)
        return statistics.fmean(dxs[-period:]) if dxs else 50.0

    def _range_efficiency(self, bars: Sequence[Dict[str, float]], window: int = 48) -> float:
        if len(bars) < window + 1:
            return 1.0
        recent = bars[-window - 1:]
        net_move = abs(float(recent[-1]["close"]) - float(recent[0]["close"]))
        path = sum(abs(float(recent[i]["close"]) - float(recent[i - 1]["close"])) for i in range(1, len(recent)))
        return net_move / max(path, self._point())

    def _atr_ratio(self, bars: Sequence[Dict[str, float]], short_period: int = 14, long_period: int = 96) -> float:
        trs = self._true_ranges(bars)
        if len(trs) < long_period:
            return 1.0
        short_atr = statistics.fmean(trs[-short_period:])
        long_atr = statistics.fmean(trs[-long_period:])
        return short_atr / max(long_atr, self._point())

    def _alternating_signal_rate(self, signals: Sequence[str]) -> float:
        directional = [sig for sig in signals if sig in {"BUY", "SELL"}]
        if len(directional) < 4:
            return 0.0
        flips = sum(1 for i in range(1, len(directional)) if directional[i] != directional[i - 1])
        return flips / max(len(directional) - 1, 1)

    def _session_label(self, ts: dt.datetime) -> str:
        if 0 <= ts.hour < 7:
            return "asia"
        if 7 <= ts.hour < 13:
            return "london_pre_us"
        if 13 <= ts.hour < 20:
            return "us_london_overlap"
        return "late_us"

    def _detect_chop(
        self,
        closed_bars: Sequence[Dict[str, float]],
        score: float,
        atr: float,
        fast_sma: float,
        slow_sma: float,
        recent_signals: Sequence[str],
    ) -> Tuple[bool, int, str]:
        if str(self.config.chop_gate).strip().lower() != "conservative_session":
            return False, 0, "disabled"
        points = 0
        reasons: List[str] = []
        adx = self._calculate_adx(closed_bars[-120:], 14)
        efficiency = self._range_efficiency(closed_bars, 48)
        ratio = self._atr_ratio(closed_bars, 14, 96)
        slope_atr = abs(fast_sma - slow_sma) / max(atr, self._point())
        alternation = self._alternating_signal_rate(recent_signals[-36:])
        if adx <= float(self.config.chop_adx_max):
            points += 1
            reasons.append("adx")
        if efficiency <= float(self.config.chop_efficiency_max):
            points += 1
            reasons.append("efficiency")
        if ratio <= float(self.config.chop_atr_ratio_max):
            points += 1
            reasons.append("atr_compression")
        if slope_atr <= float(self.config.chop_slope_atr_max):
            points += 1
            reasons.append("flat_sma")
        if alternation >= float(self.config.chop_alternation_min):
            points += 1
            reasons.append("alternating_signals")
        if abs(float(score)) <= float(self.config.chop_min_score):
            points += 1
            reasons.append("weak_score")
        return points >= int(self.config.chop_min_points), points, "+".join(reasons) or "none"

    def _chop_risk_multiplier(self, is_chop: bool, session: str) -> float:
        if str(self.config.chop_gate).strip().lower() != "conservative_session" or not is_chop:
            return 1.0
        if session == "asia":
            return 0.0
        return self._clamp(float(self.config.chop_non_asia_risk_mult), 0.0, 1.0)

    def _htf_lag_reversal_blocks(
        self,
        signal: str,
        htf_signal: str,
        close: float,
        fast_sma: float,
        slow_sma: float,
        momentum: float,
        m15_momentum: float,
        atr: float,
    ) -> bool:
        if not bool(self.config.enable_htf_lag_reversal_guard):
            return False
        buffer = max(0.0, float(self.config.htf_lag_close_sma_buffer_atr)) * max(
            float(atr),
            self._point(),
        )
        m5_threshold = max(0.0, float(self.config.htf_lag_momentum_threshold))
        m15_threshold = max(0.0, float(self.config.htf_lag_m15_threshold))
        if signal == "SELL" and htf_signal == "BEAR":
            close_reclaimed_sma = close >= max(float(fast_sma), float(slow_sma)) + buffer
            bullish_momentum = momentum >= m5_threshold or m15_momentum >= m15_threshold
            return close_reclaimed_sma and bullish_momentum
        if signal == "BUY" and htf_signal == "BULL":
            close_lost_sma = close <= min(float(fast_sma), float(slow_sma)) - buffer
            bearish_momentum = momentum <= -m5_threshold or m15_momentum <= -m15_threshold
            return close_lost_sma and bearish_momentum
        return False

    def _score_signal(
        self,
        closes: Sequence[float],
        highs: Sequence[float],
        lows: Sequence[float],
        atr: float,
        fast_sma: float,
        slow_sma: float,
        htf_fast_sma: float,
        htf_slow_sma: float,
        htf_signal: str,
        momentum: float,
        m15_momentum: float,
        spread_points: float,
        vol_ratio: float = 1.0,
        accel_score: float = 0.0,
    ) -> float:
        last_close = closes[-1]
        lookback = max(10, self.config.breakout_lookback)
        recent_high = max(highs[-lookback:])
        recent_low = min(lows[-lookback:])

        trend = 0.0
        if last_close > fast_sma > slow_sma:
            trend = 0.55
        elif last_close < fast_sma < slow_sma:
            trend = -0.55
        elif last_close > slow_sma:
            trend = 0.20
        elif last_close < slow_sma:
            trend = -0.20

        htf_bias = 0.0
        htf_gap = htf_fast_sma - htf_slow_sma
        htf_gap_strength = self._clamp(htf_gap / max(atr, self._point() * 5), -1.5, 1.5)
        htf_momentum = self._clamp(momentum, -2.0, 2.0)
        htf_bias = (
            htf_gap_strength * 0.28
            + htf_momentum * float(self.config.htf_momentum_bias_weight)
        )
        if htf_signal == "BULL":
            htf_bias = max(htf_bias, 0.22 + max(htf_gap_strength, 0.0) * 0.18)
        elif htf_signal == "BEAR":
            htf_bias = min(htf_bias, -0.22 + min(htf_gap_strength, 0.0) * 0.18)
        elif htf_signal == "NEUTRAL":
            htf_bias = 0.0
        if htf_signal == "BULL" and htf_momentum < 0:
            htf_bias *= 0.65
        elif htf_signal == "BEAR" and htf_momentum > 0:
            htf_bias *= 0.65

        breakout = 0.0
        if atr > 0:
            breakout = self._clamp((last_close - recent_high) / atr, -1.0, 1.0) * 0.30
            breakout += self._clamp((recent_low - last_close) / atr, -1.0, 1.0) * -0.30

        momentum_component = (
            self._clamp(momentum, -1.5, 1.5) * float(self.config.momentum_score_weight)
        )
        spread_penalty = 0.0
        if spread_points > max(self.config.max_spread_points, 0.0):
            spread_penalty = -0.50

        score = trend + htf_bias + breakout + momentum_component + spread_penalty

        score_dir = 1.0 if score >= 0 else -1.0
        vol_boost = min(max(vol_ratio - 1.0, 0.0), 1.0) * float(self.config.vol_ratio_weight) * score_dir
        accel_boost = min(accel_score, 1.0) * float(self.config.accel_weight) * score_dir
        score += vol_boost + accel_boost

        return self._clamp(score, -1.5, 1.5)

    def _decide_signal(self, score: float, htf_signal: str, spread_points: float) -> str:
        if spread_points > self.config.max_spread_points:
            return "NONE"
        if htf_signal == "BULL" and score >= self.config.trend_threshold:
            return "BUY"
        if htf_signal == "BEAR" and score <= -self.config.trend_threshold:
            return "SELL"
        return "NONE"

    def _quality_filter_allows(
        self,
        signal: str,
        htf_signal: str,
        compensated_htf_signal: str,
        score: float,
        adx: float,
    ) -> bool:
        if signal not in {"BUY", "SELL"}:
            return False
        min_abs_score = max(0.0, float(self.config.min_abs_score))
        if min_abs_score > 0.0 and abs(float(score)) < min_abs_score:
            return False
        min_adx = max(0.0, float(self.config.min_adx))
        if min_adx > 0.0 and float(adx) < min_adx:
            return False
        if bool(self.config.require_raw_htf_agree):
            if htf_signal != compensated_htf_signal:
                return False
            if signal == "BUY" and htf_signal != "BULL":
                return False
            if signal == "SELL" and htf_signal != "BEAR":
                return False
        return True

    def _trend_pullback_allows(
        self,
        signal: str,
        closed_bars: Sequence[Dict[str, float]],
        atr: float,
        fast_sma: float,
    ) -> bool:
        mode = str(self.config.entry_mode).strip().lower()
        if mode in {"", "immediate", "market"}:
            return True
        if mode != "pullback":
            return True
        if signal not in {"BUY", "SELL"} or not closed_bars or atr <= 0:
            return False
        last = closed_bars[-1]
        close = float(last["close"])
        high = float(last["high"])
        low = float(last["low"])
        zone = max(0.0, float(self.config.pullback_max_atr)) * float(atr)
        if signal == "BUY":
            touched_pullback_zone = low <= float(fast_sma) + zone
            closed_back_with_trend = close >= float(fast_sma)
            return touched_pullback_zone and closed_back_with_trend
        touched_pullback_zone = high >= float(fast_sma) - zone
        closed_back_with_trend = close <= float(fast_sma)
        return touched_pullback_zone and closed_back_with_trend

    def _false_breakout_reversal_signal(
        self,
        closed_bars: Sequence[Dict[str, float]],
        atr: float,
        htf_signal: str,
        spread_points: float,
    ) -> Tuple[str, str]:
        if not self.config.enable_false_breakout_reversal:
            return "NONE", "disabled"
        if spread_points > self.config.max_spread_points:
            return "NONE", "spread"
        if atr <= 0:
            return "NONE", "atr"

        lookback = max(5, int(self.config.false_breakout_lookback))
        if len(closed_bars) < lookback + 1:
            return "NONE", "bars"

        last = closed_bars[-1]
        prior = closed_bars[-lookback - 1:-1]
        prior_high = max(float(bar["high"]) for bar in prior)
        prior_low = min(float(bar["low"]) for bar in prior)
        bar_high = float(last["high"])
        bar_low = float(last["low"])
        bar_open = float(last["open"])
        bar_close = float(last["close"])
        bar_range = max(bar_high - bar_low, self._point())
        body_high = max(bar_open, bar_close)
        body_low = min(bar_open, bar_close)
        upper_wick_ratio = (bar_high - body_high) / bar_range
        lower_wick_ratio = (body_low - bar_low) / bar_range
        min_break = max(0.0, float(self.config.false_breakout_min_atr)) * atr
        close_back = max(0.0, float(self.config.false_breakout_close_back_atr)) * atr
        min_wick = max(0.0, float(self.config.false_breakout_wick_ratio))
        direction_mode = str(self.config.false_breakout_direction).strip().upper()

        upthrust = (
            bar_high >= prior_high + min_break
            and bar_close <= prior_high - close_back
            and upper_wick_ratio >= min_wick
        )
        if upthrust and direction_mode in {"SELL", "SELL_ONLY", "BOTH", "ANY"}:
            if htf_signal == "BEAR":
                return (
                    "SELL",
                    "upthrust high=%.2f prior_high=%.2f close=%.2f wick=%.2f"
                    % (bar_high, prior_high, bar_close, upper_wick_ratio),
                )
            return "NONE", "upthrust_without_bear_htf"

        spring = (
            bar_low <= prior_low - min_break
            and bar_close >= prior_low + close_back
            and lower_wick_ratio >= min_wick
        )
        if spring and direction_mode in {"BUY", "BUY_ONLY", "BOTH", "ANY"}:
            if htf_signal == "BULL":
                return (
                    "BUY",
                    "spring low=%.2f prior_low=%.2f close=%.2f wick=%.2f"
                    % (bar_low, prior_low, bar_close, lower_wick_ratio),
                )
            return "NONE", "spring_without_bull_htf"

        return "NONE", "none"

    def _handle_bar(self, snapshot: MarketSnapshot) -> None:
        positions = self._positions()
        foreign_positions = self._foreign_positions()
        if len(positions) > 0:
            self._prev_owned_positions = len(positions)
        self._log(
            "Bar %s | close=%.2f atr=%.2f fast=%.2f slow=%.2f htf=%s htf_comp=%s spread=%.1f momentum=%.2f m15_mom=%.2f score=%.2f signal=%s source=%s session=%s chop=%s chop_points=%d chop_reason=%s chop_mult=%.2f fb=%s fb_reason=%s vol_r=%.2f accel=%.2f positions=%d foreign=%d trades_today=%d losses=%d",
            snapshot.bar_time,
            snapshot.close,
            snapshot.atr,
            snapshot.fast_sma,
            snapshot.slow_sma,
            snapshot.htf_signal,
            snapshot.compensated_htf_signal,
            snapshot.spread_points,
            snapshot.momentum,
            snapshot.m15_momentum,
            snapshot.score,
            snapshot.signal,
            snapshot.signal_source,
            snapshot.session,
            snapshot.chop_is_chop,
            snapshot.chop_points,
            snapshot.chop_reason,
            snapshot.chop_risk_multiplier,
            snapshot.false_breakout_signal,
            snapshot.false_breakout_reason,
            snapshot.vol_ratio,
            snapshot.accel_score,
            len(positions),
            len(foreign_positions),
            self.state.trades_today,
            self.state.consecutive_losses,
        )
        self._recent_primary_signals.append("NONE" if snapshot.signal_source == "complement" else snapshot.signal)

        startup_warmup_active = self._record_startup_warmup_bar(snapshot)

        if positions:
            if self._maybe_profit_close(positions):
                return
            if float(self.config.auto_half_profit_usd) > 0:
                self._maybe_auto_half_close(positions)
                positions = self._positions()
            if positions and self._maybe_signal_reversal_take_profit(snapshot, positions):
                return
        else:
            changed = False
            if self.state.signal_reversal_history:
                self.state.signal_reversal_history = []
                changed = True
            if self.state.auto_half_close_done:
                self.state.auto_half_close_done = False
                changed = True
            if changed:
                self._save_state()

        if self.state.paused:
            self._log("Trading paused: %s", self.state.paused_reason)
            return

        if self._profit_pause_active(snapshot.bar_time):
            self._log("Profit-close pause blocks new entries until %s", self.state.profit_pause_until)
            return

        if self._loss_close_pause_active(snapshot.bar_time):
            self._log("Loss-close pause blocks new entries until %s", self.state.loss_pause_until)
            return

        if not self._session_allowed(snapshot.bar_time):
            self._log("Session filter blocked entry at %s UTC", snapshot.bar_time.hour)
            self._maybe_trail(snapshot, positions[0]) if positions else None
            return

        if self._friday_force_close_cutoff_reached(snapshot.bar_time):
            self._log("Friday force-close cutoff reached; block new entries at %s", snapshot.bar_time)
            self._maybe_trail(snapshot, positions[0]) if positions else None
            return

        if self.state.trades_today >= self.config.max_trades_per_day:
            self._log("Daily trade cap reached: %d", self.state.trades_today)
            self._maybe_trail(snapshot, positions[0]) if positions else None
            return

        if self.state.consecutive_losses >= self.config.max_consecutive_losses:
            if self._loss_cooldown_active():
                self._log(
                    "Loss cooldown blocks new entries: losses=%d until=%s",
                    self.state.consecutive_losses,
                    self.state.loss_cooldown_until,
                )
                self._maybe_trail(snapshot, positions[0]) if positions else None
                return
            self._log(
                "Consecutive loss cap (%d) reached but cooldown expired; resetting",
                self.state.consecutive_losses,
            )
            self.state.consecutive_losses = 0
            self._save_state()

        if not positions:
            if self._prev_owned_positions > 0 and self._deal_sync_failed:
                self._log(
                    "Deal sync pending after position close (prev=%d); blocking new entry until sync succeeds",
                    self._prev_owned_positions,
                )
                # ponytail: reset prev count so next bar isn't permanently locked.
                # Deal sync retries in the background; once bridge recovers,
                # _deal_sync_failed clears and entries resume normally.
                self._prev_owned_positions = 0
                return
            self._prev_owned_positions = 0
            if foreign_positions and not self.config.allow_foreign_positions:
                self._log("Foreign position present (%d); skip new entry.", len(foreign_positions))
                return
            if startup_warmup_active:
                return
            if snapshot.signal != "NONE" and self._cooldown_ok(snapshot.bar_time):
                self._enter(snapshot)
            return

        pos = positions[0]
        if any(self._max_hold_exceeded(open_pos, snapshot.bar_time) for open_pos in positions):
            self._log("Max holding time exceeded, closing all owned positions.")
            self.close_all_positions()
            return

        if (
            int(self.config.signal_reversal_take_profit_bars) <= 0
            and any(self._should_reverse(snapshot, open_pos) for open_pos in positions)
        ):
            self._log("Reverse signal detected, closing current owned positions first.")
            self.close_all_positions()
            time.sleep(1)
            if startup_warmup_active:
                self._log("Startup warmup active; reverse entry deferred until next completed bar.")
                return
            if not self.state.paused and self._cooldown_ok(snapshot.bar_time):
                self._enter(snapshot)
            return

        if self.config.allow_pyramiding:
            current_direction = self._direction_from_position_type(pos.type)
            if snapshot.chop_risk_multiplier < 1.0:
                self._log(
                    "Chop shield blocks pyramiding: chop_mult=%.2f signal=%s direction=%s",
                    snapshot.chop_risk_multiplier,
                    snapshot.signal,
                    current_direction,
                )
            elif (
                snapshot.signal == current_direction
                and self._cooldown_ok(snapshot.bar_time)
                and not startup_warmup_active
            ):
                self._enter(snapshot)
                return

        for open_pos in positions:
            self._maybe_trail(snapshot, open_pos)

    def _record_startup_warmup_bar(self, snapshot: MarketSnapshot) -> bool:
        warmup_bars = max(0, int(self.config.startup_warmup_bars))
        if warmup_bars <= 0:
            return False
        self._startup_bars_seen += 1
        if self._startup_bars_seen <= warmup_bars:
            self._log(
                "Startup warmup active: bar %d/%d at %s; new entries disabled until next completed bar.",
                self._startup_bars_seen,
                warmup_bars,
                snapshot.bar_time,
            )
            return True
        return False

    def _all_positions(self) -> List[PositionState]:
        raw = self.mt5.positions_get(symbol=self.symbol)
        if raw is None:
            raw, error = self._positions_once()
            if raw is None:
                if error is not None:
                    self._log("positions_get unavailable through fresh client: %s", error)
                return []
        if len(raw) == 0:
            fresh_raw, error = self._positions_once()
            if fresh_raw is None:
                if error is not None:
                    self._log("positions_get empty and fresh client unavailable: %s", error)
                return []
            raw = fresh_raw
            if len(raw) == 0:
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

    def _positions_once(self) -> Tuple[Optional[Any], Any]:
        client = MetaTrader5(host=self.config.host, port=self.config.port)
        try:
            ok = client.initialize()
            if not ok:
                return None, client.last_error()
            try:
                client.symbol_select(self.symbol, True)
            except Exception:
                pass
            positions = client.positions_get(symbol=self.symbol)
            return positions, client.last_error()
        except Exception as exc:
            return None, exc
        finally:
            try:
                client.shutdown()
            except Exception:
                pass

    def _foreign_positions(self) -> List[PositionState]:
        return [pos for pos in self._all_positions() if int(getattr(pos, "magic", 0)) != int(self.config.magic)]

    def _positions(self) -> List[PositionState]:
        return [pos for pos in self._all_positions() if int(getattr(pos, "magic", 0)) == int(self.config.magic)]

    def _should_reverse(self, snapshot: MarketSnapshot, pos: PositionState) -> bool:
        if pos.type == self.mt5.POSITION_TYPE_BUY and snapshot.signal == "SELL":
            return True
        if pos.type == self.mt5.POSITION_TYPE_SELL and snapshot.signal == "BUY":
            return True
        return False

    def _composite_signal_direction(self, snapshot: MarketSnapshot) -> str:
        if snapshot.signal in {"BUY", "SELL"}:
            return snapshot.signal
        threshold = max(float(self.config.trend_threshold), 0.0)
        if snapshot.compensated_htf_signal == "BULL" and snapshot.score >= threshold:
            return "BUY"
        if snapshot.compensated_htf_signal == "BEAR" and snapshot.score <= -threshold:
            return "SELL"
        return "NONE"

    def _maybe_signal_reversal_take_profit(
        self,
        snapshot: MarketSnapshot,
        positions: List[PositionState],
    ) -> bool:
        required_bars = int(self.config.signal_reversal_take_profit_bars)
        window_bars = int(self.config.signal_reversal_take_profit_window or required_bars)
        required_count = int(self.config.signal_reversal_take_profit_count or required_bars)
        if required_bars <= 0 or window_bars <= 0 or required_count <= 0 or not positions:
            return False
        window_bars = max(window_bars, required_count)
        direction = self._composite_signal_direction(snapshot)
        self.state.signal_reversal_history.append(direction)
        self.state.signal_reversal_history = self.state.signal_reversal_history[-max(window_bars, required_bars, 12):]
        self._save_state()

        owned_directions = {self._direction_from_position_type(pos.type) for pos in positions}
        if len(owned_directions) != 1:
            self._log(
                "Signal reversal TP monitor skip: mixed owned directions=%s history=%s",
                sorted(owned_directions),
                self.state.signal_reversal_history[-window_bars:],
            )
            return False
        owned_direction = next(iter(owned_directions))
        adverse_direction = "SELL" if owned_direction == "BUY" else "BUY"
        recent = self.state.signal_reversal_history[-window_bars:]
        adverse_count = sum(1 for item in recent if item == adverse_direction)
        if len(recent) < window_bars or adverse_count < required_count:
            self._log(
                "Signal reversal TP monitor pending: position=%s adverse=%s count=%d/%d window=%s",
                owned_direction,
                adverse_direction,
                adverse_count,
                required_count,
                recent,
            )
            return False
        direction = adverse_direction

        floating_profit = sum(float(pos.profit) for pos in positions)
        if bool(self.config.signal_reversal_profit_only) and floating_profit <= 0.0:
            self._log(
                "Signal reversal TP skipped because floating profit %.2f <= 0; adverse=%s count_window=%s",
                floating_profit,
                direction,
                recent,
            )
            return False

        self._log(
            "Signal reversal take-profit trigger: position=%s adverse=%s count=%d/%d window=%d floating_profit=%.2f history=%s",
            owned_direction,
            direction,
            adverse_count,
            required_count,
            window_bars,
            floating_profit,
            recent,
        )
        closed_ok = self.close_all_positions()
        if closed_ok:
            if floating_profit < 0.0:
                self._activate_loss_close_pause(floating_profit)
            self.state.auto_half_close_done = False
            self.state.signal_reversal_history = []
            self._save_state()
        else:
            self._log("Signal reversal take-profit did not confirm all closes.")
        return closed_ok

    def _maybe_trail_on_loop(self) -> None:
        """Check breakeven + trailing stop on every loop tick, not only on new bars.

        Intra-bar price moves can cross breakeven or trailing thresholds before
        the next M5 bar completes.  Running trail checks at loop cadence
        (default 10 s) ensures the SL is moved promptly, matching the pattern
        used by _maybe_auto_half_close_on_loop.
        """
        positions = self._positions()
        if not positions:
            return
        snapshot = self._build_snapshot()
        if snapshot is None or snapshot.atr <= 0:
            return
        for pos in positions:
            self._maybe_trail(snapshot, pos)

    def _maybe_profit_close_on_loop(self) -> None:
        """Close all owned positions once floating profit reaches a fixed USD cap.

        This is a campaign-level profit protection rule: take the whole win,
        then block fresh entries for a short timed pause instead of partially
        closing and immediately refilling exposure.
        """
        positions = self._positions()
        if positions:
            self._maybe_profit_close(positions)
            return
        self._profit_pause_active()

    def _maybe_force_close_friday(self, bar_time: dt.datetime) -> bool:
        """Close all owned positions at/after a configured Friday chart-time cutoff.

        The strategy uses MT5 chart timestamps converted via _bar_time(), so the
        flag names retain the existing UTC convention but execution is aligned
        with the same chart-time basis used by session gates.
        """
        cutoff_hour = int(self.config.force_close_friday_hour_utc)
        if cutoff_hour < 0:
            return False
        cutoff_minute = int(self.config.force_close_friday_minute_utc)
        if not (0 <= cutoff_hour <= 23 and 0 <= cutoff_minute <= 59):
            self._log(
                "Invalid Friday force-close cutoff hour=%s minute=%s; skipping.",
                cutoff_hour,
                cutoff_minute,
            )
            return False
        if not self._friday_force_close_cutoff_reached(bar_time):
            return False
        positions = self._positions()
        if not positions:
            return False
        cutoff = bar_time.replace(hour=cutoff_hour, minute=cutoff_minute, second=0, microsecond=0)
        self._log(
            "Friday force-close trigger: bar_time=%s cutoff=%s positions=%d",
            bar_time.isoformat(sep=" ", timespec="seconds"),
            cutoff.isoformat(sep=" ", timespec="seconds"),
            len(positions),
        )
        closed_ok = self.close_all_positions()
        if closed_ok:
            self.state.auto_half_close_done = False
            self.state.signal_reversal_history = []
            self._save_state()
        else:
            self._log("Friday force-close did not confirm all closes.")
        return closed_ok

    def _maybe_auto_half_close_on_loop(self) -> None:
        """Check legacy partial profit-protection on every loop.

        Kept for backward compatibility, but disabled whenever
        --auto-half-profit-usd is 0 in the active plan.
        """
        if float(self.config.auto_half_profit_usd) <= 0:
            return
        positions = self._positions()
        if positions:
            self._maybe_auto_half_close(positions)
            return
        if self.state.auto_half_close_done:
            self.state.auto_half_close_done = False
            self._save_state()

    def _trail_stable_profit_ready(self, snapshot: MarketSnapshot, pos: PositionState, profit_move: float) -> bool:
        stable_minutes = int(self.config.trail_stable_minutes)
        if stable_minutes <= 0:
            return True
        trigger_move = snapshot.atr * min(
            float(self.config.break_even_atr),
            float(self.config.trail_trigger_atr),
        )
        ticket_key = str(pos.ticket)
        if profit_move < trigger_move:
            changed = False
            if ticket_key in self.state.trail_profit_since_by_ticket:
                self.state.trail_profit_since_by_ticket.pop(ticket_key, None)
                changed = True
            if ticket_key in self.state.trailing_sl_by_ticket:
                self.state.trailing_sl_by_ticket.pop(ticket_key, None)
                self.state.trailing_direction_by_ticket.pop(ticket_key, None)
                changed = True
            if changed:
                self._save_state()
            return False
        now = self._chart_now()
        raw_since = self.state.trail_profit_since_by_ticket.get(ticket_key)
        if raw_since is None:
            self.state.trail_profit_since_by_ticket[ticket_key] = now.isoformat(timespec="seconds")
            self._save_state()
            self._log(
                "Trailing stability timer started: ticket=%s profit_move=%.2f trigger=%.2f need=%dmin",
                pos.ticket,
                profit_move,
                trigger_move,
                stable_minutes,
            )
            return False
        try:
            since = dt.datetime.fromisoformat(str(raw_since))
        except ValueError:
            self.state.trail_profit_since_by_ticket[ticket_key] = now.isoformat(timespec="seconds")
            self._save_state()
            return False
        held_minutes = (now - since).total_seconds() / 60.0
        if held_minutes < float(stable_minutes):
            self._log(
                "Trailing stability pending: ticket=%s %.1f/%d min profit_move=%.2f trigger=%.2f",
                pos.ticket,
                held_minutes,
                stable_minutes,
                profit_move,
                trigger_move,
            )
            return False
        return True

    def _remember_trailing_sl(self, pos: PositionState, sl: float) -> None:
        ticket_key = str(pos.ticket)
        self.state.trailing_sl_by_ticket[ticket_key] = float(sl)
        self.state.trailing_direction_by_ticket[ticket_key] = self._direction_from_position_type(pos.type)
        self._save_state()

    def _maybe_profit_close(self, positions: List[PositionState]) -> bool:
        threshold = float(self.config.profit_close_usd)
        pause_minutes = int(self.config.profit_close_pause_minutes)
        if threshold <= 0 or pause_minutes <= 0 or not positions:
            return False
        if self._profit_pause_active():
            return False

        floating_profit = sum(float(pos.profit) for pos in positions)
        if floating_profit < threshold:
            return False

        self._log(
            "Profit-close trigger | floating_profit=%.2f threshold=%.2f pause_minutes=%d positions=%d",
            floating_profit,
            threshold,
            pause_minutes,
            len(positions),
        )
        closed_ok = self.close_all_positions()
        if closed_ok:
            self._activate_profit_pause(floating_profit)
            self.state.auto_half_close_done = False
            self._save_state()
        else:
            self._log("Profit-close did not confirm all closes; pause not activated.")
        return closed_ok

    def _maybe_auto_half_close(self, positions: List[PositionState]) -> None:
        threshold = float(self.config.auto_half_profit_usd)
        fraction = self._clamp(float(self.config.auto_half_fraction), 0.0, 1.0)
        if threshold <= 0 or fraction <= 0 or not positions:
            return
        if self.state.auto_half_close_done:
            return

        floating_profit = sum(float(pos.profit) for pos in positions)
        if floating_profit < threshold:
            return

        self._log(
            "Auto half-close trigger | floating_profit=%.2f threshold=%.2f fraction=%.2f positions=%d",
            floating_profit,
            threshold,
            fraction,
            len(positions),
        )
        ask, bid = self._tick_prices()
        closed_any = False
        for pos in positions:
            volume = self._partial_close_volume(pos.volume, fraction)
            if volume <= 0:
                self._log("Auto half-close skip ticket=%s volume=%.2f", pos.ticket, pos.volume)
                continue
            price = bid if pos.type == self.mt5.POSITION_TYPE_BUY else ask
            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": volume,
                "type": self.mt5.ORDER_TYPE_SELL if pos.type == self.mt5.POSITION_TYPE_BUY else self.mt5.ORDER_TYPE_BUY,
                "position": pos.ticket,
                "price": price,
                "deviation": int(self.config.deviation),
                "magic": int(self.config.magic),
                "comment": "auto-half-profit",
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": self._select_filling_mode(),
            }
            if self.config.live:
                result = self._send_order_with_filling_fallback(request)
                self._log("Auto half-close result ticket=%s volume=%.2f: %s", pos.ticket, volume, self._result_to_dict(result))
                if result is not None:
                    code = getattr(result, "retcode", None)
                    if code in {self.mt5.TRADE_RETCODE_DONE, self.mt5.TRADE_RETCODE_PLACED, self.mt5.TRADE_RETCODE_DONE_PARTIAL}:
                        closed_any = True
            else:
                self._log("DRY-RUN auto half-close request: %s", request)
                closed_any = True

        if closed_any:
            self.state.auto_half_close_done = True
            self.state.last_half_close_bar_time = dt.datetime.now().isoformat(timespec="seconds")
            self._save_state()

    def _partial_close_volume(self, current_volume: float, fraction: float) -> float:
        min_vol = float(self._symbol_volume_min)
        step = float(self._symbol_volume_step or 0.01)
        target = max(0.0, current_volume * fraction)
        volume = self._round_volume(target, step)
        if volume < min_vol:
            return 0.0
        # Keep a residual position where possible; otherwise close only if the
        # current position is too small to split safely.
        if current_volume - volume < min_vol:
            residual_safe = self._round_volume(current_volume - min_vol, step)
            volume = residual_safe if residual_safe >= min_vol else current_volume
        return round(float(volume), 8)

    def _maybe_trail(self, snapshot: MarketSnapshot, pos: PositionState) -> None:
        if snapshot.atr <= 0:
            return
        ask, bid = self._tick_prices()
        current_price = bid if pos.type == self.mt5.POSITION_TYPE_BUY else ask
        profit_move = (
            current_price - pos.price_open
            if pos.type == self.mt5.POSITION_TYPE_BUY
            else pos.price_open - current_price
        )
        fee_cover_enabled = float(self.config.fee_cover_price_offset) >= 0.0
        fee_cover_offset = max(0.0, float(self.config.fee_cover_price_offset))

        if not self._trail_stable_profit_ready(snapshot, pos, profit_move):
            return

        if profit_move >= snapshot.atr * self.config.break_even_atr:
            if pos.type == self.mt5.POSITION_TYPE_BUY:
                be_sl = (
                    pos.price_open + fee_cover_offset
                    if fee_cover_enabled
                    else pos.price_open + snapshot.atr * self.config.break_even_lock_atr
                )
                if pos.sl <= 0 or be_sl > pos.sl:
                    self._modify_position(pos, sl=be_sl, tp=pos.tp)
                    self._remember_trailing_sl(pos, be_sl)
            else:
                be_sl = (
                    pos.price_open - fee_cover_offset
                    if fee_cover_enabled
                    else pos.price_open - snapshot.atr * self.config.break_even_lock_atr
                )
                if pos.sl <= 0 or be_sl < pos.sl:
                    self._modify_position(pos, sl=be_sl, tp=pos.tp)
                    self._remember_trailing_sl(pos, be_sl)

        if profit_move < snapshot.atr * self.config.trail_trigger_atr:
            return

        if pos.type == self.mt5.POSITION_TYPE_BUY:
            trail_sl = (
                pos.price_open + fee_cover_offset
                if fee_cover_enabled
                else current_price - snapshot.atr * self.config.trail_lock_atr
            )
            new_sl = max(pos.sl, trail_sl)
            if pos.sl <= 0 or new_sl > pos.sl:
                self._modify_position(pos, sl=new_sl, tp=pos.tp)
                self._remember_trailing_sl(pos, new_sl)
        else:
            base_sl = pos.sl if pos.sl > 0 else current_price + snapshot.atr * 100.0
            trail_sl = (
                pos.price_open - fee_cover_offset
                if fee_cover_enabled
                else current_price + snapshot.atr * self.config.trail_lock_atr
            )
            new_sl = min(base_sl, trail_sl)
            if pos.sl <= 0 or new_sl < pos.sl:
                self._modify_position(pos, sl=new_sl, tp=pos.tp)
                self._remember_trailing_sl(pos, new_sl)

    def _enter(self, snapshot: MarketSnapshot) -> None:
        direction = snapshot.signal
        if direction not in {"BUY", "SELL"}:
            return
        block_reason = self._new_entry_block_reason()
        if block_reason:
            self._log("New entry hard-blocked before order_send: %s", block_reason)
            return
        if self._direction_cooldown_active(direction):
            return

        ask, bid = self._tick_prices()
        price = ask if direction == "BUY" else bid
        sl, tp = self._build_sl_tp(direction, price, snapshot.atr)
        volume = self._size_position(direction, price, sl, snapshot.chop_risk_multiplier)
        if volume <= 0:
            self._log("Calculated volume is zero; skip entry.")
            return

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
        self._last_signal = direction
        self._save_state()

    def _build_sl_tp(self, direction: str, price: float, atr: float) -> Tuple[float, float]:
        sl_distance = atr * self.config.stop_atr
        primary_rr = float(self.config.primary_tp_reward_multiple)
        base_rr = primary_rr if primary_rr > 0 else float(self.config.reward_multiple)

        if self.config.enable_dynamic_tp:
            threshold = max(0.1, float(self.config.trend_threshold))
            score_strength = abs(self._last_score) / threshold
            boost = min(score_strength * 0.5, 1.0)
            reward_multiple = base_rr * (1.0 + boost)
        else:
            reward_multiple = base_rr

        tp_distance = sl_distance * reward_multiple
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

    def _size_position(self, direction: str, price: float, sl: float, risk_multiplier: float = 1.0) -> float:
        equity = self._get_equity()
        size_multiplier = self._clamp(float(risk_multiplier), 0.0, 1.0)
        if size_multiplier <= 0.0:
            self._log("Risk multiplier is zero; skip entry sizing.")
            return 0.0
        risk_amount = equity * self._effective_risk_pct() * size_multiplier
        info = self.mt5.symbol_info(self.symbol)
        if info is None:
            info, error = self._symbol_info_once()
            if info is None:
                self._log("symbol_info unavailable for sizing, using cached limits: %s", error)

        risk_per_lot = self._risk_per_lot(direction, price, sl)
        if risk_per_lot <= 0:
            raise RuntimeError("risk_per_lot invalid")

        lot_cap = float(self.config.max_lots) * size_multiplier
        owned_volume = sum(max(0.0, float(pos.volume)) for pos in self._positions())
        remaining_lot_cap = max(0.0, lot_cap - owned_volume)
        if remaining_lot_cap <= 0:
            self._log(
                "Total lot cap reached: owned=%.2f max=%.2f; skip additional entry.",
                owned_volume,
                lot_cap,
            )
            return 0.0
        per_order_cap = float(self.config.max_lots_per_order) * size_multiplier
        lot_cap = remaining_lot_cap
        if per_order_cap > 0:
            lot_cap = min(lot_cap, per_order_cap)

        raw_volume = risk_amount / risk_per_lot
        raw_volume = min(raw_volume, lot_cap)

        margin_lot = self._margin_per_lot(direction, price)
        if margin_lot > 0:
            acc, _ = self._account_info_once()
            free_margin = float(getattr(acc, "margin_free", equity) if acc is not None else equity)
            max_by_margin = (free_margin * 0.85) / margin_lot
            raw_volume = min(raw_volume, max_by_margin)

        min_vol = float(getattr(info, "volume_min", self._symbol_volume_min) if info is not None else self._symbol_volume_min)
        max_vol = float(getattr(info, "volume_max", self._symbol_volume_max) if info is not None else self._symbol_volume_max)
        raw_volume = self._clamp(raw_volume, min_vol, max_vol)
        step = float(getattr(info, "volume_step", self._symbol_volume_step) if info is not None else self._symbol_volume_step)
        volume = self._round_volume(raw_volume, step)
        return min(volume, lot_cap)

    def _effective_risk_pct(self) -> float:
        base_risk = max(0.0, float(self.config.risk_pct))
        warmup_days = max(0, int(self.config.warmup_risk_days))
        multiplier = self._clamp(float(self.config.warmup_risk_multiplier), 0.0, 1.0)
        if warmup_days <= 0 or multiplier >= 1.0:
            return base_risk
        if self._risk_warmup_active():
            effective = base_risk * multiplier
            self._log(
                "Warmup risk active: risk_pct=%.5f multiplier=%.2f effective=%.5f",
                base_risk,
                multiplier,
                effective,
            )
            return effective
        return base_risk

    def _risk_warmup_active(self) -> bool:
        warmup_days = max(0, int(self.config.warmup_risk_days))
        if warmup_days <= 0:
            return False
        started_raw = self.state.risk_warmup_started_at
        if not started_raw and self.state.current_day:
            started_raw = f"{self.state.current_day}T00:00:00"
            self.state.risk_warmup_started_at = started_raw
        if not started_raw:
            started_raw = dt.datetime.now().isoformat(timespec="seconds")
            self.state.risk_warmup_started_at = started_raw
        try:
            started = dt.datetime.fromisoformat(str(started_raw))
        except ValueError:
            self.state.risk_warmup_started_at = dt.datetime.now().isoformat(timespec="seconds")
            self._save_state()
            return True

        if self.state.current_day:
            try:
                current = dt.datetime.fromisoformat(f"{self.state.current_day}T00:00:00")
            except ValueError:
                current = dt.datetime.now()
        else:
            current = dt.datetime.now()
        return current < started + dt.timedelta(days=warmup_days)

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
            info, _ = self._symbol_info_once()
        if info is None:
            contract_size = self._symbol_contract_size
        else:
            contract_size = float(getattr(info, "trade_contract_size", self._symbol_contract_size))
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

    def _entry_order_comment(self) -> str:
        configured = str(self.config.order_comment or "").strip()
        if configured:
            return configured[:31]
        return "xauusd-trend-live" if self.config.live else "xauusd-trend-dryrun"

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
            "comment": self._entry_order_comment(),
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
            result, error = self._order_send_once(req)
            last_result = result
            self._log("order_send(type_filling=%s) -> %s", mode, self._result_to_dict(result))
            if result is None:
                self._log("order_send(type_filling=%s) failed: %s", mode, error)
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
            "comment": "trend-trail-adjust",
        }
        if self.config.live:
            result, error = self._order_send_once(request)
            self._log("SLTP modify result: %s", self._result_to_dict(result))
            if result is None:
                self._log("SLTP modify failed: %s", error)
        else:
            self._log("DRY-RUN SLTP modify: %s", request)

    def close_all_positions(self) -> bool:
        positions = self._positions()
        if not positions:
            self._log("No positions to close.")
            return True
        ask, bid = self._tick_prices()
        all_done = True
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
                "comment": "trend-close-all",
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": self._select_filling_mode(),
            }
            if self.config.live:
                result = self._send_order_with_filling_fallback(request)
                self._log("Close result: %s", self._result_to_dict(result))
                code = getattr(result, "retcode", None) if result is not None else None
                if code not in {self.mt5.TRADE_RETCODE_DONE, self.mt5.TRADE_RETCODE_PLACED, self.mt5.TRADE_RETCODE_DONE_PARTIAL}:
                    all_done = False
            else:
                self._log("DRY-RUN close request: %s", request)
        return all_done

    def _tick_prices(self) -> Tuple[float, float]:
        tick, error = self._tick_once()
        if tick is None:
            raise RuntimeError(f"symbol_info_tick failed: {error}")
        ask = float(getattr(tick, "ask", 0.0) or getattr(tick, "last", 0.0) or 0.0)
        bid = float(getattr(tick, "bid", 0.0) or getattr(tick, "last", 0.0) or 0.0)
        if ask <= 0 or bid <= 0:
            raise RuntimeError(f"Invalid bid/ask: ask={ask} bid={bid}")
        return ask, bid

    def _point(self) -> float:
        info = self.mt5.symbol_info(self.symbol)
        if info is None:
            info, _ = self._symbol_info_once()
        if info is not None:
            self._cache_symbol_info(info)
        return float(self._symbol_point or 0.01)

    def _digits(self) -> int:
        info = self.mt5.symbol_info(self.symbol)
        if info is None:
            info, _ = self._symbol_info_once()
        if info is not None:
            self._cache_symbol_info(info)
        return int(self._symbol_digits or 2)

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
            try:
                return result._asdict()
            except Exception:
                pass
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="XAUUSD trend-following MT5 strategy with hard risk rules.")
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
    parser.add_argument(
        "--max-lots-per-order",
        type=float,
        default=0.0,
        help="Maximum lots for a single entry order; 0 disables the per-order cap.",
    )
    parser.add_argument("--max-leverage", type=float, default=5.0)
    parser.add_argument("--fast-sma", type=int, default=20)
    parser.add_argument("--slow-sma", type=int, default=60)
    parser.add_argument(
        "--htf-timeframe",
        default="H1",
        choices=["M15", "M30", "H1", "H4", "D1"],
        help="Higher-timeframe trend filter timeframe. Use H4 for M30 strategy variants.",
    )
    parser.add_argument("--htf-fast-sma", type=int, default=50)
    parser.add_argument("--htf-slow-sma", type=int, default=200)
    parser.add_argument("--trend-threshold", type=float, default=0.35)
    parser.add_argument(
        "--htf-comp-momentum-threshold",
        type=float,
        default=0.75,
        help="Minimum absolute M15 momentum needed to override/compensate the raw H1 HTF filter.",
    )
    parser.add_argument(
        "--htf-momentum-bias-weight",
        type=float,
        default=0.10,
        help="Weight of short-term momentum inside the HTF bias score component.",
    )
    parser.add_argument(
        "--momentum-score-weight",
        type=float,
        default=0.25,
        help="Weight of M5 momentum in the final signal score.",
    )
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--breakout-lookback", type=int, default=20)
    parser.add_argument(
        "--enable-false-breakout-reversal",
        action="store_true",
        help="Enable sharp false-breakout reversal overlay. Default is disabled.",
    )
    parser.add_argument(
        "--false-breakout-direction",
        default="SELL_ONLY",
        choices=["SELL_ONLY", "BUY_ONLY", "BOTH", "ANY", "SELL", "BUY"],
        help="Allowed overlay direction. Research edge currently supports SELL_ONLY best.",
    )
    parser.add_argument("--false-breakout-lookback", type=int, default=20)
    parser.add_argument("--false-breakout-min-atr", type=float, default=0.15)
    parser.add_argument("--false-breakout-close-back-atr", type=float, default=0.05)
    parser.add_argument("--false-breakout-wick-ratio", type=float, default=0.45)
    parser.add_argument("--stop-atr", type=float, default=2.5)
    parser.add_argument("--reward-multiple", type=float, default=2.5)
    parser.add_argument("--trail-trigger-atr", type=float, default=1.5)
    parser.add_argument("--trail-lock-atr", type=float, default=0.5)
    parser.add_argument(
        "--trail-stable-minutes",
        type=int,
        default=0,
        help="Minutes floating profit must stay above the trail/breakeven trigger before SL is moved; 0 disables.",
    )
    parser.add_argument(
        "--trail-same-direction-cooldown-minutes",
        type=int,
        default=0,
        help="Minutes to block same-direction fresh entries after a tracked trailing SL exit; 0 disables.",
    )
    parser.add_argument("--break-even-atr", type=float, default=1.0)
    parser.add_argument("--break-even-lock-atr", type=float, default=0.15)
    parser.add_argument(
        "--fee-cover-price-offset",
        type=float,
        default=-1.0,
        help="If >=0, move SL only to entry +/- this price offset; negative uses ATR break-even/trailing behavior.",
    )
    parser.add_argument("--session-start-utc", type=int, default=6)
    parser.add_argument("--session-end-utc", type=int, default=21)
    parser.add_argument("--max-spread-points", type=float, default=120.0)
    parser.add_argument("--max-trades-per-day", type=int, default=4)
    parser.add_argument("--max-consecutive-losses", type=int, default=3)
    parser.add_argument(
        "--loss-cooldown-losses",
        type=int,
        default=0,
        help="Consecutive closed losses required to block new entries temporarily; 0 disables.",
    )
    parser.add_argument(
        "--loss-cooldown-minutes",
        type=int,
        default=0,
        help="Minutes to block new entries after loss-cooldown trigger; 0 disables.",
    )
    parser.add_argument(
        "--profit-close-usd",
        type=float,
        default=0.0,
        help="If owned floating profit reaches this USD value, close all owned positions; 0 disables.",
    )
    parser.add_argument(
        "--profit-close-pause-minutes",
        type=int,
        default=0,
        help="Deprecated: minutes to block fresh entries after --profit-close-usd closes positions; use 0 to disable profit-close pauses.",
    )
    parser.add_argument(
        "--loss-close-pause-minutes",
        type=int,
        default=0,
        help="Minutes to block fresh entries after any losing close; 0 disables.",
    )
    parser.add_argument(
        "--auto-half-profit-usd",
        type=float,
        default=0.0,
        help="If owned floating profit reaches this USD value, close a fraction of each owned position once per position campaign; 0 disables.",
    )
    parser.add_argument(
        "--auto-half-fraction",
        type=float,
        default=0.5,
        help="Fraction of each owned position to close when --auto-half-profit-usd triggers.",
    )
    parser.add_argument(
        "--half-close-cooldown-bars",
        type=int,
        default=0,
        help="Bars to block fresh entries after auto half-close so the strategy does not refill exposure immediately.",
    )
    parser.add_argument(
        "--warmup-risk-days",
        type=int,
        default=0,
        help="Calendar days after baseline seed to reduce position risk; 0 disables.",
    )
    parser.add_argument(
        "--warmup-risk-multiplier",
        type=float,
        default=1.0,
        help="Multiplier applied to --risk-pct during --warmup-risk-days, e.g. 0.5 for half risk.",
    )
    parser.add_argument(
        "--primary-tp-reward-multiple",
        type=float,
        default=0.0,
        help="If >0, use this closer initial TP RR while keeping reward-multiple as the campaign reference.",
    )
    parser.add_argument("--cooldown-bars-after-trade", type=int, default=2)
    parser.add_argument(
        "--startup-warmup-bars",
        type=int,
        default=1,
        help="Completed bars to observe after process start/restart before allowing new entries.",
    )
    parser.add_argument("--max-hold-minutes", type=int, default=0)
    parser.add_argument("--loop-seconds", type=int, default=10)
    parser.add_argument("--lookback-bars", type=int, default=200)
    parser.add_argument("--htf-lookback-bars", type=int, default=400)
    parser.add_argument("--max-concentration-share", type=float, default=0.45)
    parser.add_argument("--min-positive-days-for-concentration", type=int, default=3)
    parser.add_argument(
        "--chop-gate",
        default="none",
        choices=["none", "conservative_session"],
        help="Chop handling mode. conservative_session pauses trend entries in Asia chop and scales non-Asia chop risk.",
    )
    parser.add_argument("--chop-adx-max", type=float, default=18.0)
    parser.add_argument("--chop-efficiency-max", type=float, default=0.18)
    parser.add_argument("--chop-atr-ratio-max", type=float, default=0.85)
    parser.add_argument("--chop-slope-atr-max", type=float, default=1.00)
    parser.add_argument("--chop-alternation-min", type=float, default=0.55)
    parser.add_argument("--chop-min-score", type=float, default=0.65)
    parser.add_argument("--chop-min-points", type=int, default=3)
    parser.add_argument("--chop-non-asia-risk-mult", type=float, default=0.25)
    parser.add_argument(
        "--min-abs-score",
        type=float,
        default=0.0,
        help="Block trend entries unless abs(score) is at least this value; 0 disables.",
    )
    parser.add_argument(
        "--min-adx",
        type=float,
        default=0.0,
        help="Block trend entries unless M5 ADX(14) is at least this value; 0 disables.",
    )
    parser.add_argument(
        "--require-raw-htf-agree",
        action="store_true",
        help="Require raw HTF direction to match compensated HTF and the entry direction.",
    )
    parser.add_argument(
        "--entry-mode",
        default="immediate",
        choices=["immediate", "pullback"],
        help="Trend-entry trigger mode. pullback waits for a touch near the fast SMA before entering.",
    )
    parser.add_argument(
        "--pullback-max-atr",
        type=float,
        default=0.35,
        help="For --entry-mode pullback, max ATR distance from fast SMA that counts as a retest.",
    )
    parser.add_argument(
        "--enable-htf-lag-reversal-guard",
        action="store_true",
        help="Block fresh trend entries when the H1 bias appears stale versus a sharp M5/M15 reversal.",
    )
    parser.add_argument("--htf-lag-momentum-threshold", type=float, default=0.70)
    parser.add_argument("--htf-lag-m15-threshold", type=float, default=0.50)
    parser.add_argument("--htf-lag-close-sma-buffer-atr", type=float, default=0.05)
    parser.add_argument(
        "--signal-reversal-take-profit-bars",
        type=int,
        default=0,
        help="Close owned positions after this many consecutive adverse composite signals; 0 keeps legacy one-bar reverse behavior.",
    )
    parser.add_argument(
        "--signal-reversal-take-profit-window",
        type=int,
        default=0,
        help="Rolling window size for adverse composite-signal take-profit; 0 uses --signal-reversal-take-profit-bars.",
    )
    parser.add_argument(
        "--signal-reversal-take-profit-count",
        type=int,
        default=0,
        help="Required adverse composite-signal count within the rolling window; 0 uses --signal-reversal-take-profit-bars.",
    )
    parser.add_argument(
        "--signal-reversal-profit-only",
        action="store_true",
        help="Only execute signal-reversal take-profit when owned floating P/L is positive.",
    )
    parser.add_argument(
        "--signal-reversal-pause-minutes",
        type=int,
        default=0,
        help="Minutes to block fresh entries after signal-reversal take-profit closes positions; 0 disables.",
    )
    parser.add_argument(
        "--force-close-friday-hour-utc",
        type=int,
        default=-1,
        help="Friday chart/UTC hour to close all owned positions; -1 disables.",
    )
    parser.add_argument(
        "--force-close-friday-minute-utc",
        type=int,
        default=0,
        help="Minute within --force-close-friday-hour-utc for Friday owned-position close.",
    )
    parser.add_argument(
        "--vol-lookback",
        type=int,
        default=48,
        help="Bars for baseline ATR computation used in vol-ratio scoring.",
    )
    parser.add_argument(
        "--vol-ratio-weight",
        type=float,
        default=0.15,
        help="Score boost weight for vol-ratio (>1 = high vol increases directional confidence).",
    )
    parser.add_argument(
        "--accel-weight",
        type=float,
        default=0.10,
        help="Score boost weight for acceleration (momentum change strength).",
    )
    parser.add_argument(
        "--enable-dynamic-tp",
        action="store_true",
        help="Scale TP reward-multiple by score strength for stronger entries.",
    )
    parser.add_argument(
        "--enable-proactive-ipc",
        action="store_true",
        help="Periodically shutdown/reinit MT5 client to prevent IPC degradation.",
    )
    parser.add_argument(
        "--proactive-ipc-interval",
        type=int,
        default=30,
        help="Number of fetch_bars calls between proactive IPC refreshes.",
    )
    parser.add_argument(
        "--allow-pyramiding",
        action="store_true",
        help="Allow same-direction add-on entries while owned positions exist, bounded by max-lots and per-order cap.",
    )
    parser.add_argument(
        "--allow-foreign-positions",
        action="store_true",
        help="Allow new strategy entries even if same-symbol positions with a different magic number exist.",
    )
    parser.add_argument("--state-path", default=STATE_PATH_DEFAULT)
    parser.add_argument("--log-file", default=LOG_PATH_DEFAULT)
    parser.add_argument(
        "--terminal-path",
        default=r"C:\Program Files\MetaTrader 5\terminal64.exe",
        help="Windows terminal path inside the Wine prefix. Set empty to skip explicit initialize(path=...).",
    )
    parser.add_argument("--deviation", type=int, default=30)
    parser.add_argument("--magic", type=int, default=204494)
    parser.add_argument(
        "--order-comment",
        default="",
        help="MT5 entry order comment; use distinct ASCII labels per sleeve for position attribution.",
    )
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
        max_lots_per_order=float(args.max_lots_per_order),
        max_leverage=float(args.max_leverage),
        fast_sma=int(args.fast_sma),
        slow_sma=int(args.slow_sma),
        htf_timeframe=str(args.htf_timeframe),
        htf_fast_sma=int(args.htf_fast_sma),
        htf_slow_sma=int(args.htf_slow_sma),
        trend_threshold=float(args.trend_threshold),
        htf_comp_momentum_threshold=float(args.htf_comp_momentum_threshold),
        htf_momentum_bias_weight=float(args.htf_momentum_bias_weight),
        momentum_score_weight=float(args.momentum_score_weight),
        atr_period=int(args.atr_period),
        breakout_lookback=int(args.breakout_lookback),
        enable_false_breakout_reversal=bool(args.enable_false_breakout_reversal),
        false_breakout_direction=str(args.false_breakout_direction),
        false_breakout_lookback=int(args.false_breakout_lookback),
        false_breakout_min_atr=float(args.false_breakout_min_atr),
        false_breakout_close_back_atr=float(args.false_breakout_close_back_atr),
        false_breakout_wick_ratio=float(args.false_breakout_wick_ratio),
        stop_atr=float(args.stop_atr),
        reward_multiple=float(args.reward_multiple),
        trail_trigger_atr=float(args.trail_trigger_atr),
        trail_lock_atr=float(args.trail_lock_atr),
        trail_stable_minutes=int(args.trail_stable_minutes),
        trail_same_direction_cooldown_minutes=int(args.trail_same_direction_cooldown_minutes),
        break_even_atr=float(args.break_even_atr),
        break_even_lock_atr=float(args.break_even_lock_atr),
        fee_cover_price_offset=float(args.fee_cover_price_offset),
        session_start_utc=int(args.session_start_utc),
        session_end_utc=int(args.session_end_utc),
        max_spread_points=float(args.max_spread_points),
        max_trades_per_day=int(args.max_trades_per_day),
        max_consecutive_losses=int(args.max_consecutive_losses),
        loss_cooldown_losses=int(args.loss_cooldown_losses),
        loss_cooldown_minutes=int(args.loss_cooldown_minutes),
        profit_close_usd=float(args.profit_close_usd),
        profit_close_pause_minutes=int(args.profit_close_pause_minutes),
        loss_close_pause_minutes=int(args.loss_close_pause_minutes),
        auto_half_profit_usd=float(args.auto_half_profit_usd),
        auto_half_fraction=float(args.auto_half_fraction),
        half_close_cooldown_bars=int(args.half_close_cooldown_bars),
        warmup_risk_days=int(args.warmup_risk_days),
        warmup_risk_multiplier=float(args.warmup_risk_multiplier),
        primary_tp_reward_multiple=float(args.primary_tp_reward_multiple),
        cooldown_bars_after_trade=int(args.cooldown_bars_after_trade),
        startup_warmup_bars=int(args.startup_warmup_bars),
        max_hold_minutes=int(args.max_hold_minutes),
        loop_seconds=int(args.loop_seconds),
        lookback_bars=int(args.lookback_bars),
        htf_lookback_bars=int(args.htf_lookback_bars),
        max_concentration_share=float(args.max_concentration_share),
        min_positive_days_for_concentration=int(args.min_positive_days_for_concentration),
        chop_gate=str(args.chop_gate),
        chop_adx_max=float(args.chop_adx_max),
        chop_efficiency_max=float(args.chop_efficiency_max),
        chop_atr_ratio_max=float(args.chop_atr_ratio_max),
        chop_slope_atr_max=float(args.chop_slope_atr_max),
        chop_alternation_min=float(args.chop_alternation_min),
        chop_min_score=float(args.chop_min_score),
        chop_min_points=int(args.chop_min_points),
        chop_non_asia_risk_mult=float(args.chop_non_asia_risk_mult),
        min_abs_score=float(args.min_abs_score),
        min_adx=float(args.min_adx),
        require_raw_htf_agree=bool(args.require_raw_htf_agree),
        entry_mode=str(args.entry_mode),
        pullback_max_atr=float(args.pullback_max_atr),
        enable_htf_lag_reversal_guard=bool(args.enable_htf_lag_reversal_guard),
        htf_lag_momentum_threshold=float(args.htf_lag_momentum_threshold),
        htf_lag_m15_threshold=float(args.htf_lag_m15_threshold),
        htf_lag_close_sma_buffer_atr=float(args.htf_lag_close_sma_buffer_atr),
        signal_reversal_take_profit_bars=int(args.signal_reversal_take_profit_bars),
        signal_reversal_take_profit_window=int(args.signal_reversal_take_profit_window),
        signal_reversal_take_profit_count=int(args.signal_reversal_take_profit_count),
        signal_reversal_profit_only=bool(args.signal_reversal_profit_only),
        signal_reversal_pause_minutes=int(args.signal_reversal_pause_minutes),
        force_close_friday_hour_utc=int(args.force_close_friday_hour_utc),
        force_close_friday_minute_utc=int(args.force_close_friday_minute_utc),
        vol_lookback=int(args.vol_lookback),
        vol_ratio_weight=float(args.vol_ratio_weight),
        accel_weight=float(args.accel_weight),
        enable_dynamic_tp=bool(args.enable_dynamic_tp),
        enable_proactive_ipc=bool(args.enable_proactive_ipc),
        proactive_ipc_interval=int(args.proactive_ipc_interval),
        allow_pyramiding=bool(args.allow_pyramiding),
        allow_foreign_positions=bool(args.allow_foreign_positions),
        state_path=args.state_path,
        log_file=args.log_file,
        terminal_path=terminal_path,
        deviation=int(args.deviation),
        magic=int(args.magic),
        order_comment=str(args.order_comment),
        log_level=args.log_level,
    )

    strategy = XAUUSDTrendStrategy(config)
    strategy.run()


if __name__ == "__main__":
    main()
