#!/usr/bin/env python3.14
"""XAUUSD momentum surfer — opencode 做的. Catches explosive intraday moves with aggressive trailing.

Core idea:
- M1 scalper detecting momentum acceleration (velocity of velocity).
- Enter when short-term momentum is building rapidly.
- Ultra-tight trailing stop locks profits fast.
- Win rate does NOT matter; a few big winners dominate return.

Risk:
- Per-trade risk 1% of equity.
- Hard campaign drawdown limit 5%.
- 10-day campaign target.
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


LOGGER = logging.getLogger("xauusd_momentum_surfer")
STATE_PATH_DEFAULT = "/home/chain4655/Documents/Sample/Python/xauusd_momentum_surfer_state.json"
LOG_PATH_DEFAULT = "/home/chain4655/Documents/Sample/Python/xauusd_momentum_surfer.log"
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
    atr_period: int
    vol_lookback: int
    mom_lookback: int
    accel_min: float
    entry_buffer_atr: float
    stop_atr: float
    reward_multiple: float
    trail_trigger_atr: float
    trail_lock_atr: float
    resonance_enabled: bool
    resonance_compression_atr: float
    resonance_sweep_lookback: int
    resonance_reclaim_body_min_atr: float
    max_spread_points: float
    max_trades_per_day: int
    max_consecutive_losses: int
    loss_cooldown_losses: int
    loss_cooldown_minutes: int
    loss_close_pause_minutes: int
    profit_close_usd: float
    profit_close_pause_minutes: int
    cooldown_seconds: int
    max_hold_minutes: int
    loop_seconds: int
    lookback_bars: int

    state_path: str
    log_file: str
    terminal_path: Optional[str]
    deviation: int
    magic: int
    order_comment: str
    log_level: str


@dataclasses.dataclass
class MarketSnapshot:
    bar_time: dt.datetime
    close: float
    high: float
    low: float
    atr: float
    baseline_atr: float
    vol_ratio: float
    mom_1: float
    mom_2: float
    mom_3: float
    accel: float
    accel_score: float
    spread_points: float
    signal: str
    signal_strength: float



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
    loss_cooldown_until: Optional[str] = None
    loss_cooldown_triggered_at: Optional[str] = None
    last_trade_time: Optional[str] = None
    last_processed_deal_time: Optional[str] = None
    last_processed_deal_ticket: int = 0
    last_close_profit: float = 0.0
    loss_pause_until: Optional[str] = None
    profit_pause_until: Optional[str] = None
    paused_reason: str = ""
    paused: bool = False


class XAUUSDMomentumSurferStrategy:
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
        self._equity_fail_streak = 0
        self._rates_fail_streak = 0
        self._connect_fail_streak = 0
        self._last_reconnect_ts = 0.0
        self._cached_symbol_info = None
        self._cached_point: Optional[float] = None
        self._cached_digits: Optional[int] = None
        self._initialized = False

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
                        "MomentumSurfer started: %s %s live=%s",
                        self.symbol,
                        self.config.timeframe,
                        self.config.live,
                    )
                    self._initialized = True

                self._ensure_day_context(snapshot.bar_time)
                self._sync_closed_trades()
                self._risk_guard(snapshot)
                self._maybe_profit_close_on_loop()

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
                loss_cooldown_until=data.get("loss_cooldown_until"),
                loss_cooldown_triggered_at=data.get("loss_cooldown_triggered_at"),
                last_trade_time=data.get("last_trade_time"),
                last_processed_deal_time=data.get("last_processed_deal_time"),
                last_processed_deal_ticket=int(data.get("last_processed_deal_ticket", 0) or 0),
                last_close_profit=float(data.get("last_close_profit", 0.0)),
                loss_pause_until=data.get("loss_pause_until"),
                profit_pause_until=data.get("profit_pause_until"),
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
            "loss_cooldown_until": self.state.loss_cooldown_until,
            "loss_cooldown_triggered_at": self.state.loss_cooldown_triggered_at,
            "last_trade_time": self.state.last_trade_time,
            "last_processed_deal_time": self.state.last_processed_deal_time,
            "last_processed_deal_ticket": self.state.last_processed_deal_ticket,
            "last_close_profit": self.state.last_close_profit,
            "loss_pause_until": self.state.loss_pause_until,
            "profit_pause_until": self.state.profit_pause_until,
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
        """Return naive MT5 chart/server time for strategy-state comparisons.

        FTMO/pymt5linux bar/deal/tick epochs are on the broker chart-time basis
        (currently real UTC+3), not the Asia/Taipei host clock. Cooldowns,
        sessions, and state timestamps must all use the same chart/server basis.
        `history_deals_get()` query windows intentionally stay host-local in
        `_history_query_now()` because the bridge expects local naive datetimes.
        """
        try:
            tick, _error = self._tick_once()
            tick_time = int(getattr(tick, "time", 0) or 0) if tick is not None else 0
            if tick_time > 0:
                return self._mt5_timestamp_to_chart_time(tick_time)
        except Exception:
            pass
        # Fallback only for pre-connect/test paths; live paths should use MT5 tick time.
        return dt.datetime.now(dt.UTC).replace(tzinfo=None)

    def _history_query_now(self) -> dt.datetime:
        """Naive local timestamp expected by Wine/pymt5linux history_deals_get."""
        return dt.datetime.now()

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

    def _is_connected(self) -> bool:
        try:
            ti = self.mt5.terminal_info()
            ai = self.mt5.account_info()
            return ti is not None and ai is not None
        except Exception:
            return False

    def _symbol_info(self) -> Any:
        info, _error = self._symbol_info_once()
        if info is not None:
            self._cached_symbol_info = info
            point = float(getattr(info, "point", 0.0) or 0.0)
            digits = int(getattr(info, "digits", 0) or 0)
            if point > 0:
                self._cached_point = point
            if digits >= 0:
                self._cached_digits = digits
            return info
        if self._cached_symbol_info is not None:
            return self._cached_symbol_info
        raise RuntimeError("symbol_info unavailable")

    def _prepare_symbol(self) -> None:
        for attempt in range(5):
            selected = self.mt5.symbol_select(self.symbol, True)
            info = self.mt5.symbol_info(self.symbol) if selected else None
            if info is not None:
                self._cached_symbol_info = info
                self._cached_point = float(getattr(info, "point", 0.0) or 0.0)
                self._cached_digits = int(getattr(info, "digits", 0) or 0)
                break
            self._log(
                "symbol_select/info unavailable (attempt %s/5): selected=%s error=%s",
                attempt + 1,
                selected,
                self.mt5.last_error(),
            )
            time.sleep(1)
        else:
            raise RuntimeError(f"symbol_info unavailable after retries: {self.mt5.last_error()}")
        self._log(
            "Symbol ready: %s | digits=%s point=%s volume_min=%s volume_step=%s",
            self.symbol,
            getattr(info, "digits", "?"),
            getattr(info, "point", "?"),
            getattr(info, "volume_min", "?"),
            getattr(info, "volume_step", "?"),
        )

    def _shutdown(self) -> None:
        try:
            self.mt5.shutdown()
        except Exception:
            pass

    # ---- Fresh-client pattern (ported from mt5_xauusd_trend_strategy.py) ----
    # On this Wine/MT5 bridge, long-lived pymt5linux clients degrade into
    # (-10004, 'No IPC connection') after repeated market-data / trading calls.
    # Each heavy operation gets its own short-lived client so the stale-client
    # path never triggers, while self.mt5 stays for misc / constant lookups.

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

    def _copy_rates_once(self, timeframe: int, bars: int) -> Tuple[Optional[Any], Any]:
        """Fetch rates through a short-lived pymt5linux client."""
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

    def _history_deals_once(self, from_time: dt.datetime, to_time: dt.datetime) -> Tuple[Optional[Any], Any]:
        client = MetaTrader5(host=self.config.host, port=self.config.port)
        try:
            ok = client.initialize()
            if not ok:
                return None, client.last_error()
            deals = client.history_deals_get(from_time, to_time)
            return deals, client.last_error()
        except Exception as exc:
            return None, exc
        finally:
            try:
                client.shutdown()
            except Exception:
                pass

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
            return info, client.last_error()
        except Exception as exc:
            return None, exc
        finally:
            try:
                client.shutdown()
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
        max_attempts = 5
        for attempt in range(max_attempts):
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
                "account_info unavailable (attempt %s/%s, streak=%s): %s",
                attempt + 1,
                max_attempts,
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
        raise RuntimeError(f"account_info unavailable after reconnect attempts: {self.mt5.last_error()}")

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

    def _cooldown_active(self, now: dt.datetime) -> bool:
        until = self.state.loss_cooldown_until
        if not until:
            return False
        try:
            until_dt = dt.datetime.fromisoformat(until)
        except Exception:
            return False
        if now < until_dt:
            remaining = until_dt - now
            self._log(
                "Loss cooldown active until %s | remaining=%s | consecutive_losses=%d",
                until_dt.isoformat(sep=' '),
                str(remaining).split('.')[0],
                self.state.consecutive_losses,
            )
            return True
        self.state.loss_cooldown_until = None
        self.state.loss_cooldown_triggered_at = None
        self._save_state()
        self._log("Loss cooldown expired at %s", now.isoformat(sep=' '))
        return False

    def _trigger_loss_cooldown(self, now: dt.datetime) -> None:
        cooldown_until = now + dt.timedelta(minutes=int(self.config.loss_cooldown_minutes))
        self.state.loss_cooldown_triggered_at = now.isoformat()
        self.state.loss_cooldown_until = cooldown_until.isoformat()
        self._save_state()
        self._log(
            "Loss cooldown triggered: consecutive_losses=%d threshold=%d until=%s",
            self.state.consecutive_losses,
            self.config.loss_cooldown_losses,
            cooldown_until.isoformat(sep=' '),
        )

    def _maybe_handle_loss_cooldown(self, now: dt.datetime) -> None:
        if self.state.consecutive_losses >= int(self.config.loss_cooldown_losses):
            if not self.state.loss_cooldown_until:
                self._trigger_loss_cooldown(now)
            return
        self._cooldown_active(now)

    def _pause_until_active(self, raw_until: Optional[str], label: str, now: Optional[dt.datetime] = None) -> bool:
        if not raw_until:
            return False
        try:
            until = dt.datetime.fromisoformat(str(raw_until))
        except Exception:
            return False
        current = now or self._chart_now()
        if current < until:
            self._log("%s active until %s", label, until.isoformat(sep=" "))
            return True
        return False

    def _loss_close_pause_active(self, now: Optional[dt.datetime] = None) -> bool:
        if self._pause_until_active(self.state.loss_pause_until, "Loss-close pause", now):
            return True
        if self.state.loss_pause_until:
            self.state.loss_pause_until = None
            self._save_state()
        return False

    def _profit_close_pause_active(self, now: Optional[dt.datetime] = None) -> bool:
        if self._pause_until_active(self.state.profit_pause_until, "Profit-close pause", now):
            return True
        if self.state.profit_pause_until:
            self.state.profit_pause_until = None
            self._save_state()
        return False

    def _activate_loss_close_pause(self, close_profit: float, closed_at: Optional[dt.datetime] = None) -> None:
        minutes = int(self.config.loss_close_pause_minutes)
        if minutes <= 0 or close_profit >= 0:
            return
        base = closed_at or self._chart_now()
        until = base + dt.timedelta(minutes=minutes)
        self.state.loss_pause_until = until.isoformat(timespec="seconds")
        self._save_state()
        self._log("Loss-close pause triggered: profit=%.2f until=%s", close_profit, self.state.loss_pause_until)

    def _activate_profit_close_pause(self) -> None:
        minutes = int(self.config.profit_close_pause_minutes)
        if minutes <= 0:
            return
        until = self._chart_now() + dt.timedelta(minutes=minutes)
        self.state.profit_pause_until = until.isoformat(timespec="seconds")
        self._save_state()
        self._log("Profit-close pause triggered until=%s", self.state.profit_pause_until)

    def _risk_guard(self, snapshot: MarketSnapshot) -> None:
        equity = self._get_equity()
        metrics = self.account_metrics.update(equity, snapshot.bar_time.date().isoformat())
        self.state.current_day = metrics.current_day
        self.state.last_equity = metrics.equity
        self.state.initial_equity = metrics.initial_equity
        self.state.day_start_equity = metrics.day_start_equity
        self.state.max_equity_seen = metrics.max_equity_seen

        daily_dd = metrics.daily_dd
        total_dd = metrics.total_dd
        profit_progress = metrics.profit_pct

        self._log(
            "Risk | equity=%.2f shared_day_start=%.2f shared_initial=%.2f shared_peak=%.2f day_dd=%.2f%% total_dd=%.2f%% profit=%.2f%% paused=%s",
            metrics.equity,
            metrics.day_start_equity,
            metrics.initial_equity,
            metrics.max_equity_seen,
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
        need = max(self.config.atr_period, self.config.vol_lookback, self.config.mom_lookback) + 10
        bars = self._fetch_bars(min_count=need)
        if len(bars) < need:
            raise RuntimeError(
                f"Not enough bars: got={len(bars)} need={need}"
            )

        last_closed = bars[-2]
        closes = [bar["close"] for bar in bars[:-1]]
        highs = [bar["high"] for bar in bars[:-1]]
        lows = [bar["low"] for bar in bars[:-1]]

        bar_time = self._bar_time(last_closed)
        atr = self._atr(bars[:-1], self.config.atr_period)
        baseline_period = min(self.config.vol_lookback, self.config.atr_period)
        baseline_offset = min(max(0, self.config.mom_lookback), len(bars) - baseline_period - 2)
        baseline_atr = self._atr(bars[:-1 - baseline_offset], baseline_period)
        vol_ratio = atr / max(baseline_atr, 1e-9)

        mom_1 = closes[-1] - closes[-2] if len(closes) >= 2 else 0.0
        mom_2 = closes[-2] - closes[-3] if len(closes) >= 3 else 0.0
        mom_3 = closes[-3] - closes[-4] if len(closes) >= 4 else 0.0
        accel = mom_1 - mom_2

        accel_score = self._score_accel(mom_1, mom_2, mom_3, accel, atr, vol_ratio)
        spread_points = self._spread_points()
        if self.config.resonance_enabled:
            signal, strength, _resonance_meta = self._decide_resonance_signal(
                bars[:-1],
                atr=atr,
                spread_points=spread_points,
                max_spread_points=self.config.max_spread_points,
                compression_atr=self.config.resonance_compression_atr,
                sweep_lookback=self.config.resonance_sweep_lookback,
                reclaim_body_min_atr=self.config.resonance_reclaim_body_min_atr,
            )
            accel_score = max(accel_score, strength) if signal != "NONE" else strength
        else:
            signal, strength = self._decide_signal(mom_1, mom_2, accel, accel_score, atr, vol_ratio)

        return MarketSnapshot(
            bar_time=bar_time,
            close=float(last_closed["close"]),
            high=float(last_closed["high"]),
            low=float(last_closed["low"]),
            atr=atr,
            baseline_atr=baseline_atr,
            vol_ratio=vol_ratio,
            mom_1=mom_1,
            mom_2=mom_2,
            mom_3=mom_3,
            accel=accel,
            accel_score=accel_score,
            spread_points=spread_points,
            signal=signal,
            signal_strength=strength,

        )

    def _fetch_bars(self, min_count: Optional[int] = None) -> List[Dict[str, float]]:
        fetch_count = max(int(self.config.lookback_bars), int(min_count or 0))
        max_attempts = 5
        for attempt in range(max_attempts):
            rates, error = self._copy_rates_once(self.timeframe, fetch_count)
            if rates is not None:
                self._rates_fail_streak = 0
                return [self._normalize_bar(row) for row in list(rates)]

            self._rates_fail_streak += 1
            self._log(
                "copy_rates_from_pos unavailable (attempt %s/%s, streak=%s): %s",
                attempt + 1,
                max_attempts,
                self._rates_fail_streak,
                error,
            )
            time.sleep(1)
            if self._rates_fail_streak >= 3:
                self._handle_transient_ipc(RuntimeError("copy_rates_from_pos unavailable"))
                self._rates_fail_streak = 0
                self._connect_fail_streak = 0

        raise RuntimeError(f"copy_rates_from_pos failed after {max_attempts} attempts: {self.mt5.last_error()}")

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
            return 0.0
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

    def _score_accel(self, mom_1: float, mom_2: float, mom_3: float, accel: float, atr: float, vol_ratio: float) -> float:
        if atr <= 0:
            return 0.0

        accel_norm = abs(accel) / max(atr, 1e-9)
        mom_alignment = 0.0
        if mom_1 > 0 and mom_2 > 0 and mom_3 > 0:
            mom_alignment = 0.3
        elif mom_1 < 0 and mom_2 < 0 and mom_3 < 0:
            mom_alignment = 0.3

        mom_strength = (abs(mom_1) / max(atr, 1e-9)) * 0.3
        vol_boost = max(0.0, (vol_ratio - 1.0)) * 0.4

        score = accel_norm + mom_alignment + mom_strength + vol_boost
        return score

    def _decide_signal(self, mom_1: float, mom_2: float, accel: float, accel_score: float, atr: float, vol_ratio: float) -> Tuple[str, float]:
        spread = self._spread_points()
        if spread > self.config.max_spread_points:
            return "NONE", 0.0
        if vol_ratio < 1.0:
            return "NONE", 0.0
        if accel_score < self.config.accel_min:
            return "NONE", 0.0

        needed = mom_1 > 0 and mom_2 > 0 and accel > 0
        if needed and accel_score >= self.config.accel_min:
            return "BUY", accel_score
        needed = mom_1 < 0 and mom_2 < 0 and accel < 0
        if needed and accel_score >= self.config.accel_min:
            return "SELL", accel_score

        return "NONE", accel_score

    def _decide_resonance_signal(
        self,
        bars: Sequence[Dict[str, float]],
        atr: float,
        spread_points: float,
        max_spread_points: float,
        compression_atr: float,
        sweep_lookback: int,
        reclaim_body_min_atr: float,
    ) -> Tuple[str, float, Dict[str, Any]]:
        meta: Dict[str, Any] = {
            "trend": "NONE",
            "swept_zone": False,
            "reclaimed": False,
            "compressed": False,
            "ema9": 0.0,
            "ema21": 0.0,
            "ema50": 0.0,
            "avwap": 0.0,
        }
        if spread_points > max_spread_points or atr <= 0 or len(bars) < 55:
            return "NONE", 0.0, meta

        closes = [float(bar["close"]) for bar in bars]
        ema9 = self._ema(closes, 9)
        ema21 = self._ema(closes, 21)
        ema50 = self._ema(closes, 50)
        avwap = self._session_avwap(bars)
        meta.update({"ema9": ema9, "ema21": ema21, "ema50": ema50, "avwap": avwap})

        if ema9 > ema21 > ema50:
            meta["trend"] = "UP"
        elif ema9 < ema21 < ema50:
            meta["trend"] = "DOWN"
        else:
            return "NONE", 0.0, meta

        # ponytail: AVWAP is logged as context; EMA9/21 is the actionable reclaim zone.
        zone_low = min(ema9, ema21)
        zone_high = max(ema9, ema21)
        zone_width = zone_high - zone_low
        compressed = zone_width <= atr * compression_atr
        meta["compressed"] = compressed
        if not compressed:
            return "NONE", 0.0, meta

        reversal = bars[-1]
        signal_window = bars[-2:]
        lookback_start = max(0, len(bars) - int(sweep_lookback) - 1)
        previous = bars[lookback_start:-1]
        recent_low = min(float(bar["low"]) for bar in previous) if previous else float(reversal["low"])
        recent_high = max(float(bar["high"]) for bar in previous) if previous else float(reversal["high"])
        sweep_low = min(float(bar["low"]) for bar in signal_window)
        sweep_high = max(float(bar["high"]) for bar in signal_window)
        open_price = float(reversal["open"])
        close = float(reversal["close"])
        high = float(reversal["high"])
        low = float(reversal["low"])
        body = abs(close - open_price)
        min_body = atr * reclaim_body_min_atr

        if meta["trend"] == "UP":
            swept_zone = sweep_low <= zone_low or sweep_low <= recent_low
            reclaimed = close > open_price and close > zone_high and body >= min_body
            meta["swept_zone"] = swept_zone
            meta["reclaimed"] = reclaimed
            if swept_zone and reclaimed:
                strength = self._resonance_strength(close, zone_width, atr, body, zone_high, zone_low)
                return "BUY", strength, meta

        if meta["trend"] == "DOWN":
            swept_zone = sweep_high >= zone_high or sweep_high >= recent_high
            reclaimed = close < open_price and close < zone_low and body >= min_body
            meta["swept_zone"] = swept_zone
            meta["reclaimed"] = reclaimed
            if swept_zone and reclaimed:
                strength = self._resonance_strength(close, zone_width, atr, body, zone_high, zone_low)
                return "SELL", strength, meta

        return "NONE", 0.0, meta

    def _ema(self, values: Sequence[float], period: int) -> float:
        if not values:
            return 0.0
        if len(values) < period:
            return float(values[-1])
        alpha = 2.0 / (float(period) + 1.0)
        ema = statistics.fmean(float(value) for value in values[:period])
        for value in values[period:]:
            ema = (float(value) * alpha) + (ema * (1.0 - alpha))
        return float(ema)

    def _session_avwap(self, bars: Sequence[Dict[str, float]]) -> float:
        if not bars:
            return 0.0
        anchor_day = self._bar_time(bars[-1]).date()
        numerator = 0.0
        denominator = 0.0
        for bar in bars:
            if self._bar_time(bar).date() != anchor_day:
                continue
            typical = (float(bar["high"]) + float(bar["low"]) + float(bar["close"])) / 3.0
            volume = max(float(bar.get("tick_volume", 0.0)), 1.0)
            numerator += typical * volume
            denominator += volume
        if denominator <= 0:
            return float(bars[-1]["close"])
        return numerator / denominator

    def _resonance_strength(
        self,
        close: float,
        zone_width: float,
        atr: float,
        body: float,
        zone_high: float,
        zone_low: float,
    ) -> float:
        compression_score = max(0.0, 1.0 - (zone_width / max(atr * 3.0, 1e-9)))
        reclaim_distance = min(abs(close - ((zone_high + zone_low) / 2.0)) / max(atr, 1e-9), 2.0)
        body_score = min(body / max(atr, 1e-9), 2.0)
        return 1.0 + compression_score + reclaim_distance + body_score

    def _handle_bar(self, snapshot: MarketSnapshot) -> None:
        all_positions = self._all_positions()
        positions = [pos for pos in all_positions if int(pos.magic) == int(self.config.magic)]
        foreign_positions = [pos for pos in all_positions if int(pos.magic) != int(self.config.magic)]
        all_position_lots = self._position_lots(all_positions)
        foreign_position_lots = self._position_lots(foreign_positions)

        self._log(
            "Bar %s | close=%.2f atr=%.2f vol_ratio=%.2f accel=%.4f score=%.2f signal=%s strength=%.2f  spread=%.1f positions=%d foreign=%d lots=%.2f foreign_lots=%.2f trades_today=%d losses=%d",
            snapshot.bar_time,
            snapshot.close,
            snapshot.atr,
            snapshot.vol_ratio,
            snapshot.accel,
            snapshot.accel_score,
            snapshot.signal,
            snapshot.signal_strength,
            snapshot.spread_points,
            len(positions),
            len(foreign_positions),
            all_position_lots,
            foreign_position_lots,
            self.state.trades_today,
            self.state.consecutive_losses,
        )

        if self.state.paused:
            self._log("Trading paused: %s", self.state.paused_reason)
            return

        if self._loss_close_pause_active(snapshot.bar_time) or self._profit_close_pause_active(snapshot.bar_time):
            for pos in positions:
                self._trail_position(snapshot, pos)
            return

        if self._cooldown_active(snapshot.bar_time):
            if positions:
                self._trail_position(snapshot, positions[0])
            return

        if self.state.trades_today >= self.config.max_trades_per_day:
            self._log("Daily trade cap reached: %d", self.state.trades_today)
            if positions:
                self._trail_position(snapshot, positions[0])
            return

        if self.state.consecutive_losses >= self.config.max_consecutive_losses:
            self._log("Consecutive loss cap reached: %d", self.state.consecutive_losses)
            if positions:
                self._trail_position(snapshot, positions[0])
            return

        for pos in positions:
            if self._max_hold_exceeded(pos, snapshot.bar_time):
                self._log("Max holding time exceeded, closing position.")
                self.close_all_positions()
                return
            self._trail_position(snapshot, pos)

        if all_position_lots >= float(self.config.max_lots):
            self._log(
                "Open position lots %.2f >= max_lots %.2f; skip new entry but signal remains %s.",
                all_position_lots,
                self.config.max_lots,
                snapshot.signal,
            )
            return

        if snapshot.signal != "NONE" and self._cooldown_ok(snapshot.bar_time):
            if 0 < time.time() - self._last_reconnect_ts < 60:
                self._log("Recent MT5 reconnect; skip new entry until bridge is stable for 60s.")
                return
            self._enter(snapshot)

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

    def _position_lots(self, positions: List[PositionState]) -> float:
        return sum(abs(float(pos.volume)) for pos in positions)

    def _foreign_positions(self) -> List[PositionState]:
        return [pos for pos in self._all_positions() if int(getattr(pos, "magic", 0)) != int(self.config.magic)]

    def _positions(self) -> List[PositionState]:
        return [pos for pos in self._all_positions() if int(getattr(pos, "magic", 0)) == int(self.config.magic)]

    def _trail_position(self, snapshot: MarketSnapshot, pos: PositionState) -> None:
        if snapshot.atr <= 0:
            return
        ask, bid = self._tick_prices()
        current_price = bid if pos.type == self.mt5.POSITION_TYPE_BUY else ask
        profit_move = (current_price - pos.price_open) if pos.type == self.mt5.POSITION_TYPE_BUY else (pos.price_open - current_price)

        if profit_move >= snapshot.atr * self.config.trail_trigger_atr:
            if pos.type == self.mt5.POSITION_TYPE_BUY:
                new_sl = max(pos.sl, current_price - snapshot.atr * self.config.trail_lock_atr)
                if pos.sl <= 0 or new_sl > pos.sl:
                    self._modify_position(pos, sl=new_sl, tp=pos.tp)
            else:
                new_sl = current_price + snapshot.atr * self.config.trail_lock_atr
                if pos.sl > 0:
                    new_sl = min(pos.sl, new_sl)
                if pos.sl <= 0 or new_sl < pos.sl:
                    self._modify_position(pos, sl=new_sl, tp=pos.tp)

    def _enter(self, snapshot: MarketSnapshot) -> None:
        direction = snapshot.signal
        if direction not in {"BUY", "SELL"}:
            return

        ask, bid = self._tick_prices()
        price = ask if direction == "BUY" else bid
        sl, tp = self._build_sl_tp(direction, price, snapshot.atr)
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
            snapshot.signal_strength,
        )

        if self.config.live:
            result = self._send_order_with_filling_fallback(request)
            self._log("order_send result: %s", self._result_to_dict(result))
            if result is None:
                raise RuntimeError(f"order_send failed: {self.mt5.last_error()}")
        else:
            self._log("DRY-RUN request: %s", request)

        self.state.trades_today += 1
        self.state.last_trade_time = snapshot.bar_time.isoformat()
        self._save_state()

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
        info, error = self._symbol_info_once()
        if info is None:
            # Fallback to cached symbol info
            info = self._cached_symbol_info
            if info is None:
                raise RuntimeError(f"symbol_info unavailable: {error}")

        risk_per_lot = self._risk_per_lot(direction, price, sl)
        if risk_per_lot <= 0:
            raise RuntimeError("risk_per_lot invalid")

        raw_volume = risk_amount / risk_per_lot
        raw_volume = min(raw_volume, self.config.max_lots, self.config.max_lots_per_order)

        margin_lot = self._margin_per_lot(direction, price)
        if margin_lot > 0:
            acc, _error = self._account_info_once()
            free_margin = float(getattr(acc, "margin_free", equity) if acc is not None else equity)
            max_by_margin = (free_margin * 0.85) / margin_lot
            raw_volume = min(raw_volume, max_by_margin)

        min_vol = float(getattr(info, "volume_min", 0.01))
        step = float(getattr(info, "volume_step", 0.01))
        raw_volume = max(raw_volume, min_vol)
        volume = self._round_volume(raw_volume, step)
        return min(volume, self.config.max_lots, self.config.max_lots_per_order)

    def _risk_per_lot(self, direction: str, price: float, sl: float) -> float:
        order_type = self.mt5.ORDER_TYPE_BUY if direction == "BUY" else self.mt5.ORDER_TYPE_SELL
        try:
            value = self.mt5.order_calc_profit(order_type, self.symbol, 1.0, price, sl)
            if value is not None:
                return abs(float(value))
        except Exception:
            pass
        info, _error = self._symbol_info_once()
        if info is None:
            info = self._cached_symbol_info
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
            "comment": self._entry_order_comment(),
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }

    def _entry_order_comment(self) -> str:
        configured = str(self.config.order_comment or "").strip()
        if configured:
            return configured[:31]
        return ("xauusd-surfer-live" if self.config.live else "xauusd-surfer-dryrun")[:31]

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
            result, _error = self._order_send_once(req)
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
            "comment": "surfer-trail",
        }
        if self.config.live:
            result, error = self._order_send_once(request)
            self._log("SLTP modify result: %s", self._result_to_dict(result))
            if result is None:
                self._log("SLTP modify failed: %s", error)
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
                "comment": "surfer-close-all",
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": self._select_filling_mode(),
            }
            if self.config.live:
                result = self._send_order_with_filling_fallback(request)
                self._log("Close result: %s", self._result_to_dict(result))
            else:
                self._log("DRY-RUN close request: %s", request)

    def _maybe_profit_close_on_loop(self) -> None:
        threshold = float(self.config.profit_close_usd)
        if threshold <= 0:
            return
        positions = self._positions()
        if not positions:
            return
        floating_profit = sum(float(pos.profit) for pos in positions)
        if floating_profit < threshold:
            return
        self._log("Profit-close threshold hit: floating_profit=%.2f threshold=%.2f", floating_profit, threshold)
        self.close_all_positions()
        self._activate_profit_close_pause()

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
        if self._cached_point and self._cached_point > 0:
            return float(self._cached_point)
        info = self._symbol_info()
        return float(getattr(info, "point", 0.0) or 0.0)

    def _digits(self) -> int:
        if self._cached_digits is not None:
            return int(self._cached_digits)
        info = self._symbol_info()
        return int(getattr(info, "digits", 0) or 0)

    def _spread_points(self) -> float:
        tick, error = self._tick_once()
        if tick is None:
            raise RuntimeError(f"symbol_info_tick failed: {error}")
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
            try:
                return result._asdict()
            except Exception:
                pass
        return result

    def _cooldown_ok(self, bar_time: dt.datetime) -> bool:
        if self.state.last_trade_time is None:
            return True
        try:
            last_dt = dt.datetime.fromisoformat(str(self.state.last_trade_time))
        except Exception:
            return True
        elapsed = (bar_time - last_dt).total_seconds()
        return elapsed >= float(self.config.cooldown_seconds)

    def _max_hold_exceeded(self, pos: PositionState, bar_time: dt.datetime) -> bool:
        if int(self.config.max_hold_minutes) <= 0:
            return False
        if pos.time_open is None:
            return False
        # MT5 bar timestamps and position timestamps can be in different timezone
        # conventions under Wine/pymt5linux. Use monotonic wall-clock age for hold time
        # so a fresh position is not closed immediately by chart/server time skew.
        held_minutes = max(0.0, (time.time() - float(pos.time_open)) / 60.0)
        return held_minutes >= float(self.config.max_hold_minutes)

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
        return dt.datetime.fromtimestamp(int(timestamp), dt.UTC).replace(tzinfo=None)

    def _sync_closed_trades(self) -> None:
        to_time = self._history_query_now()
        from_time = to_time - dt.timedelta(days=30)
        try:
            deals, error = self._history_deals_once(from_time, to_time)
            if deals is None and error is not None:
                # Fallback to long-lived client
                deals = self.mt5.history_deals_get(from_time, to_time)
        except Exception as exc:
            self._log("Closed deal sync failed: %s", exc)
            return
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
            deal_iso = self._mt5_timestamp_to_chart_time(int(deal_time)).isoformat()
            if latest_seen is not None:
                if deal_iso < latest_seen:
                    continue
                if deal_iso == latest_seen and deal_ticket <= latest_seen_ticket:
                    continue
            if not self._is_owned_closing_deal(deal):
                continue
            closed_at = dt.datetime.fromisoformat(deal_iso)
            net_profit = self._deal_net_profit(deal)
            self.state.last_close_profit = net_profit
            self.state.last_processed_deal_time = deal_iso
            self.state.last_processed_deal_ticket = deal_ticket
            latest_seen = deal_iso
            latest_seen_ticket = deal_ticket
            if abs(net_profit) < REALIZED_PNL_NOISE_USD:
                self._log(
                    "Closed deal sync ignored noise | time=%s ticket=%s symbol=%s magic=%s net=%.2f consecutive_losses=%d",
                    deal_iso,
                    deal_ticket,
                    getattr(deal, "symbol", ""),
                    getattr(deal, "magic", ""),
                    net_profit,
                    self.state.consecutive_losses,
                )
                continue
            if net_profit < 0:
                self.state.consecutive_losses += 1
                if self.state.consecutive_losses >= int(self.config.loss_cooldown_losses):
                    self._trigger_loss_cooldown(closed_at)
                self._activate_loss_close_pause(net_profit, closed_at)
            elif net_profit > 0:
                self.state.consecutive_losses = 0
                self.state.loss_cooldown_until = None
                self.state.loss_cooldown_triggered_at = None
            self._log(
                "Closed deal sync | time=%s ticket=%s symbol=%s magic=%s net=%.2f consecutive_losses=%d",
                deal_iso,
                deal_ticket,
                getattr(deal, "symbol", ""),
                getattr(deal, "magic", ""),
                net_profit,
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
        keywords = (
            "broken pipe",
            "connection reset",
            "transport",
            "ipc",
            "no ipc connection",
            "stream has been closed",
            "timeout",
            "dead object",
            "symbol_info unavailable",
            "account_info unavailable",
            "copy_rates_from_pos failed",
        )
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
        # pymt5linux client objects can remain poisoned after EOFError/No IPC connection.
        # Recreate the client before initialize(), rather than reusing the closed stream.
        self.mt5 = MetaTrader5(host=self.config.host, port=self.config.port)
        self.timeframe = self._resolve_timeframe(self.config.timeframe)
        time.sleep(1)
        self._connect()
        self._prepare_symbol()
        self._connect_fail_streak = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="XAUUSD Momentum Surfer — catch explosive intraday moves.")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18812)
    parser.add_argument("--live", action="store_true", help="Actually send orders.")
    parser.add_argument("--start-equity", type=float, default=10000.0)
    parser.add_argument("--daily-dd-limit", type=float, default=0.03)
    parser.add_argument("--total-dd-limit", type=float, default=0.05)
    parser.add_argument("--profit-target", type=float, default=0.15)
    parser.add_argument("--risk-pct", type=float, default=0.01)
    parser.add_argument("--max-lots", type=float, default=3.0)
    parser.add_argument("--max-lots-per-order", type=float, default=3.0)
    parser.add_argument("--max-leverage", type=float, default=10.0)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--vol-lookback", type=int, default=50)
    parser.add_argument("--mom-lookback", type=int, default=3)
    parser.add_argument("--accel-min", type=float, default=0.80)
    parser.add_argument("--entry-buffer-atr", type=float, default=0.10)
    parser.add_argument("--stop-atr", type=float, default=1.80)
    parser.add_argument("--reward-multiple", type=float, default=3.0)
    parser.add_argument("--trail-trigger-atr", type=float, default=0.60)
    parser.add_argument("--trail-lock-atr", type=float, default=0.25)
    parser.add_argument(
        "--resonance-enabled",
        action="store_true",
        help="Use Martin-style EMA9/21/50 + session AVWAP confluence sweep/reclaim signals.",
    )
    parser.add_argument("--resonance-compression-atr", type=float, default=2.5)
    parser.add_argument("--resonance-sweep-lookback", type=int, default=12)
    parser.add_argument("--resonance-reclaim-body-min-atr", type=float, default=0.25)
    parser.add_argument("--max-spread-points", type=float, default=150.0)
    parser.add_argument("--max-trades-per-day", type=int, default=10)
    parser.add_argument("--max-consecutive-losses", type=int, default=5)
    parser.add_argument("--loss-cooldown-losses", type=int, default=3)
    parser.add_argument("--loss-cooldown-minutes", type=int, default=15)
    parser.add_argument("--loss-close-pause-minutes", type=int, default=0)
    parser.add_argument("--profit-close-usd", type=float, default=0.0)
    parser.add_argument("--profit-close-pause-minutes", type=int, default=0)
    parser.add_argument("--cooldown-seconds", type=int, default=30)
    parser.add_argument("--max-hold-minutes", type=int, default=120)
    parser.add_argument("--loop-seconds", type=int, default=10)
    parser.add_argument("--lookback-bars", type=int, default=120)
    parser.add_argument("--state-path", default=STATE_PATH_DEFAULT)
    parser.add_argument("--log-file", default=LOG_PATH_DEFAULT)
    parser.add_argument(
        "--terminal-path",
        default=r"C:\Program Files\MetaTrader 5\terminal64.exe",
        help="Windows terminal path inside the Wine prefix. Set empty to skip explicit initialize(path=...).",
    )
    parser.add_argument("--deviation", type=int, default=30)
    parser.add_argument("--magic", type=int, default=210511)
    parser.add_argument("--order-comment", default="")
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
        atr_period=int(args.atr_period),
        vol_lookback=int(args.vol_lookback),
        mom_lookback=int(args.mom_lookback),
        accel_min=float(args.accel_min),
        entry_buffer_atr=float(args.entry_buffer_atr),
        stop_atr=float(args.stop_atr),
        reward_multiple=float(args.reward_multiple),
        trail_trigger_atr=float(args.trail_trigger_atr),
        trail_lock_atr=float(args.trail_lock_atr),
        resonance_enabled=bool(args.resonance_enabled),
        resonance_compression_atr=float(args.resonance_compression_atr),
        resonance_sweep_lookback=int(args.resonance_sweep_lookback),
        resonance_reclaim_body_min_atr=float(args.resonance_reclaim_body_min_atr),
        max_spread_points=float(args.max_spread_points),
        max_trades_per_day=int(args.max_trades_per_day),
        max_consecutive_losses=int(args.max_consecutive_losses),
        loss_cooldown_losses=int(args.loss_cooldown_losses),
        loss_cooldown_minutes=int(args.loss_cooldown_minutes),
        loss_close_pause_minutes=int(args.loss_close_pause_minutes),
        profit_close_usd=float(args.profit_close_usd),
        profit_close_pause_minutes=int(args.profit_close_pause_minutes),
        cooldown_seconds=int(args.cooldown_seconds),
        max_hold_minutes=int(args.max_hold_minutes),
        loop_seconds=int(args.loop_seconds),
        lookback_bars=int(args.lookback_bars),
        state_path=args.state_path,
        log_file=args.log_file,
        terminal_path=terminal_path,
        deviation=int(args.deviation),
        magic=int(args.magic),
        order_comment=str(args.order_comment),
        log_level=args.log_level,
    )

    strategy = XAUUSDMomentumSurferStrategy(config)
    strategy.run()


if __name__ == "__main__":
    main()
