#!/usr/bin/env python3.14
"""Live XAUUSD Tide-Wave Grid strategy.

Ported from tide_wave_grid_backtest.py. Conservative live variant:
- Tide: EMA fast/slow regime filter.
- Wave: RSI + z-score distance from EMA center.
- Grid: ATR-spaced same-direction basket with basket TP / center reversion / hard stop.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from pymt5linux import MetaTrader5  # type: ignore[import-not-found]
except ImportError:
    sys.path.insert(0, "/home/chain4655/.openharness-venv/lib64/python3.14/site-packages")
    from pymt5linux import MetaTrader5  # type: ignore[import-not-found]

ROOT = Path("/home/chain4655/Documents/Projects/MT5")
sys.path.insert(0, str(ROOT))

from tide_wave_grid_backtest import (  # noqa: E402
    Bar,
    TideWaveGridConfig,
    allowed_side,
    atr_series,
    classify_regime,
    ema_series,
    lot_for_level,
    rolling_z,
    rsi_series,
)
from shared.account_metrics import AccountMetricsStore  # noqa: E402

LOGGER = logging.getLogger("xauusd_tide_wave_grid")


@dataclass(frozen=True)
class LiveConfig:
    symbol: str
    timeframe: str
    host: str
    port: int
    live: bool
    loop_seconds: int
    lookback_bars: int
    max_spread_points: float
    max_daily_loss_pct: float
    max_total_dd_pct: float
    max_foreign_lots: float
    state_path: str
    log_file: str
    terminal_path: Optional[str]
    deviation: int
    magic: int
    log_level: str
    grid: TideWaveGridConfig


@dataclass
class Position:
    ticket: int
    type: int
    volume: float
    price_open: float
    price_current: float
    profit: float
    magic: int
    comment: str
    time: int


@dataclass
class Snapshot:
    bar_time: dt.datetime
    close: float
    atr: float
    center: float
    regime: str
    z: float
    rsi: float
    step: float
    signal: str
    spread_points: float


class TideWaveGridLive:
    def __init__(self, config: LiveConfig) -> None:
        self.config = config
        self.mt5 = MetaTrader5(host=config.host, port=config.port)
        self.log_file = Path(config.log_file)
        self.state_path = Path(config.state_path)
        self.account_metrics = AccountMetricsStore()
        self.state: Dict[str, Any] = self._load_state()
        self._last_bar_time: Optional[dt.datetime] = None
        self._digits_cache: Optional[int] = None
        self._point_cache: Optional[float] = None

    def run(self) -> None:
        while True:
            try:
                self._connect()
                self._prepare_symbol()
                break
            except Exception as exc:
                self._log("Startup connect error: %s; retrying in %ss", exc, self.config.loop_seconds)
                self._shutdown()
                time.sleep(self.config.loop_seconds)
        self._log("TideWaveGrid started: %s %s live=%s", self.config.symbol, self.config.timeframe, self.config.live)
        while True:
            try:
                snapshot = self._build_snapshot()
                self._risk_guard(snapshot)
                positions = self._positions()
                foreign = self._foreign_positions()
                total_lots = sum(pos.volume for pos in positions)
                foreign_lots = sum(pos.volume for pos in foreign)
                self._log(
                    "Bar %s | close=%.2f atr=%.2f center=%.2f regime=%s z=%.2f rsi=%.1f signal=%s spread=%.1f positions=%d lots=%.2f foreign=%d foreign_lots=%.2f",
                    snapshot.bar_time,
                    snapshot.close,
                    snapshot.atr,
                    snapshot.center,
                    snapshot.regime,
                    snapshot.z,
                    snapshot.rsi,
                    snapshot.signal,
                    snapshot.spread_points,
                    len(positions),
                    total_lots,
                    len(foreign),
                    foreign_lots,
                )
                self._manage_basket(snapshot, positions)
                if self._last_bar_time is not None and snapshot.bar_time <= self._last_bar_time:
                    time.sleep(self.config.loop_seconds)
                    continue
                self._last_bar_time = snapshot.bar_time
                if not positions:
                    self._maybe_enter(snapshot, foreign_lots)
                time.sleep(self.config.loop_seconds)
            except KeyboardInterrupt:
                self._log("Interrupted; shutting down")
                break
            except Exception as exc:
                self._log("Loop error: %s", exc)
                try:
                    self._reconnect()
                except Exception as reconnect_exc:
                    self._log("Reconnect failed: %s; retrying in %ss", reconnect_exc, self.config.loop_seconds)
                    self._shutdown()
                time.sleep(self.config.loop_seconds)
        self._shutdown()

    def _load_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {"paused": False, "paused_reason": ""}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._log("State load failed, starting fresh: %s", exc)
            return {"paused": False, "paused_reason": ""}

    def _save_state(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        except OSError as exc:
            self._log("State save failed: %s", exc)

    def _log(self, message: str, *args: Any) -> None:
        line = message % args if args else message
        LOGGER.info(line)
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(f"{dt.datetime.now().isoformat(sep=' ', timespec='seconds')} {line}\n")
        except OSError:
            pass

    def _connect(self) -> None:
        if self.config.terminal_path:
            ok = self.mt5.initialize(path=self.config.terminal_path)
        else:
            ok = self.mt5.initialize()
        self._log("initialize() -> %s | last_error=%s", ok, self.mt5.last_error())
        if not ok:
            raise RuntimeError(f"MT5 initialize failed: {self.mt5.last_error()}")

    def _reconnect(self) -> None:
        self._shutdown()
        time.sleep(1)
        self._connect()
        self._prepare_symbol()

    def _shutdown(self) -> None:
        try:
            self.mt5.shutdown()
        except Exception:
            pass

    def _prepare_symbol(self) -> None:
        if not self.mt5.symbol_select(self.config.symbol, True):
            raise RuntimeError(f"symbol_select failed: {self.mt5.last_error()}")
        info = self.mt5.symbol_info(self.config.symbol)
        if info is None:
            raise RuntimeError("symbol_info unavailable")
        self._digits_cache = int(getattr(info, "digits", 2))
        self._point_cache = float(getattr(info, "point", 0.01))
        self._log("Symbol ready: %s digits=%s point=%s", self.config.symbol, self._digits_cache, self._point_cache)

    def _timeframe_const(self) -> int:
        mapping = {
            "M1": self.mt5.TIMEFRAME_M1,
            "M5": self.mt5.TIMEFRAME_M5,
            "M15": self.mt5.TIMEFRAME_M15,
            "H1": self.mt5.TIMEFRAME_H1,
        }
        key = self.config.timeframe.upper()
        if key not in mapping:
            raise ValueError(f"Unsupported timeframe: {self.config.timeframe}")
        return mapping[key]

    def _fetch_bars(self) -> List[Bar]:
        # Work around Wine/pymt5linux persistent-client market-data flaps.
        self._shutdown()
        time.sleep(0.2)
        self._connect()
        self.mt5.symbol_select(self.config.symbol, True)
        rates = self.mt5.copy_rates_from_pos(self.config.symbol, self._timeframe_const(), 0, self.config.lookback_bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"copy_rates_from_pos failed: {self.mt5.last_error()}")
        bars = [
            Bar(
                time=dt.datetime.fromtimestamp(int(row["time"])),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                tick_volume=float(row["tick_volume"]),
            )
            for row in rates
        ]
        bars.sort(key=lambda bar: bar.time)
        return bars

    def _build_snapshot(self) -> Snapshot:
        bars = self._fetch_bars()
        cfg = self.config.grid
        need = max(cfg.ema_tide_slow + 5, cfg.z_window + 5, cfg.atr_period + 5, cfg.rsi_period + 5)
        if len(bars) < need:
            raise RuntimeError(f"not enough bars: got={len(bars)} need={need}")
        closes = [bar.close for bar in bars]
        center = ema_series(closes, cfg.ema_center)
        tide_fast = ema_series(closes, cfg.ema_tide_fast)
        tide_slow = ema_series(closes, cfg.ema_tide_slow)
        atr_vals = atr_series(bars, cfg.atr_period)
        rsi_vals = rsi_series(closes, cfg.rsi_period)
        z_vals = rolling_z(closes, center, cfg.z_window)
        i = len(bars) - 2
        bar = bars[i]
        if center[i] is None or tide_fast[i] is None or tide_slow[i] is None:
            raise RuntimeError("indicator warmup incomplete")
        if atr_vals[i] is None or rsi_vals[i] is None or z_vals[i] is None:
            raise RuntimeError("indicator warmup incomplete")
        atr = float(atr_vals[i])
        center_value = float(center[i])
        regime = classify_regime(bar.close, float(tide_fast[i]), float(tide_slow[i]), atr, cfg)
        z = float(z_vals[i])
        rsi = float(rsi_vals[i])
        signal = "NONE"
        if z <= -cfg.z_entry and rsi <= cfg.rsi_buy:
            signal = "BUY"
        elif z >= cfg.z_entry and rsi >= cfg.rsi_sell:
            signal = "SELL"
        if signal != "NONE" and not allowed_side(regime, cfg, signal):
            signal = "NONE"
        return Snapshot(
            bar_time=bar.time,
            close=bar.close,
            atr=atr,
            center=center_value,
            regime=regime,
            z=z,
            rsi=rsi,
            step=max(atr * cfg.grid_step_atr, self._point() * 20),
            signal=signal,
            spread_points=self._spread_points(),
        )

    def _risk_guard(self, snapshot: Snapshot) -> None:
        account = self.mt5.account_info()
        if account is None:
            raise RuntimeError(f"account_info unavailable: {self.mt5.last_error()}")
        equity = float(getattr(account, "equity", 0.0))
        metrics = self.account_metrics.update(equity, snapshot.bar_time.date().isoformat())
        daily_dd = metrics.daily_dd
        total_dd = metrics.total_dd
        self.state.update(
            {
                "current_day": metrics.current_day,
                "last_equity": metrics.equity,
                "initial_equity": metrics.initial_equity,
                "day_start_equity": metrics.day_start_equity,
                "max_equity_seen": metrics.max_equity_seen,
            }
        )
        self._save_state()
        self._log(
            "Risk | equity=%.2f shared_day_start=%.2f shared_initial=%.2f shared_peak=%.2f day_dd=%.2f%% total_dd=%.2f%% paused=%s",
            metrics.equity,
            metrics.day_start_equity,
            metrics.initial_equity,
            metrics.max_equity_seen,
            daily_dd * 100.0,
            total_dd * 100.0,
            self.state.get("paused", False),
        )
        if daily_dd >= self.config.max_daily_loss_pct:
            self.state["paused"] = True
            self.state["paused_reason"] = f"Daily drawdown {daily_dd * 100.0:.2f}% >= {self.config.max_daily_loss_pct * 100.0:.2f}%"
            self._save_state()
            self.close_all_positions()
            raise SystemExit(1)
        if total_dd >= self.config.max_total_dd_pct:
            self.state["paused"] = True
            self.state["paused_reason"] = f"Total drawdown {total_dd * 100.0:.2f}% >= {self.config.max_total_dd_pct * 100.0:.2f}%"
            self._save_state()
            self.close_all_positions()
            raise SystemExit(1)

    def _positions_raw(self) -> List[Any]:
        raw = self.mt5.positions_get(symbol=self.config.symbol)
        if not raw:
            return []
        return list(raw)

    def _positions(self) -> List[Position]:
        return [self._position_from_raw(pos) for pos in self._positions_raw() if int(getattr(pos, "magic", 0) or 0) == self.config.magic]

    def _foreign_positions(self) -> List[Position]:
        return [self._position_from_raw(pos) for pos in self._positions_raw() if int(getattr(pos, "magic", 0) or 0) != self.config.magic]

    def _position_from_raw(self, pos: Any) -> Position:
        return Position(
            ticket=int(getattr(pos, "ticket", 0)),
            type=int(getattr(pos, "type", 0)),
            volume=float(getattr(pos, "volume", 0.0)),
            price_open=float(getattr(pos, "price_open", 0.0)),
            price_current=float(getattr(pos, "price_current", 0.0)),
            profit=float(getattr(pos, "profit", 0.0)),
            magic=int(getattr(pos, "magic", 0) or 0),
            comment=str(getattr(pos, "comment", "") or ""),
            time=int(getattr(pos, "time", 0) or 0),
        )

    def _manage_basket(self, snapshot: Snapshot, positions: Sequence[Position]) -> None:
        if not positions:
            return
        side = "BUY" if positions[0].type == self.mt5.POSITION_TYPE_BUY else "SELL"
        if any(("BUY" if pos.type == self.mt5.POSITION_TYPE_BUY else "SELL") != side for pos in positions):
            self._log("Mixed basket detected; no grid add/exit management")
            return
        avg_entry = self._weighted_avg_position(positions)
        total_volume = sum(pos.volume for pos in positions)
        mult = 1.0 if side == "BUY" else -1.0
        favorable_atr = (snapshot.close - avg_entry) * mult / max(snapshot.atr, 1e-9)
        adverse_atr = -favorable_atr
        target = avg_entry + mult * max(self.config.grid.take_profit_atr * snapshot.atr, self._point() * 30)
        center_exit_ok = (snapshot.close - snapshot.center) * mult >= -self.config.grid.basket_center_exit_atr * snapshot.atr
        reason = ""
        if favorable_atr >= self.config.grid.take_profit_atr:
            reason = "BASKET_TP"
        elif center_exit_ok and (len(positions) >= 2 or favorable_atr >= 0):
            reason = "CENTER_REVERT"
        elif adverse_atr >= self.config.grid.hard_stop_atr:
            reason = "HARD_STOP"
        oldest_time = min(pos.time for pos in positions) if positions else 0
        if oldest_time:
            max_hold_seconds = self.config.grid.max_hold_bars * self._tf_seconds()
            if time.time() - oldest_time >= max_hold_seconds:
                reason = reason or "TIME_STOP"
        if reason:
            self._log("Basket exit %s side=%s positions=%d lots=%.2f avg=%.2f target=%.2f fav_atr=%.2f adverse_atr=%.2f", reason, side, len(positions), total_volume, avg_entry, target, favorable_atr, adverse_atr)
            self.close_positions(positions)
            self.state["last_basket_exit_time"] = dt.datetime.now().isoformat()
            self._save_state()
            return
        if len(positions) < self.config.grid.max_levels:
            distance = (avg_entry - snapshot.close) if side == "BUY" else (snapshot.close - avg_entry)
            next_level = len(positions) + 1
            if distance >= snapshot.step * next_level:
                lot = lot_for_level(self.config.grid, next_level, total_volume)
                if lot >= 0.01:
                    self._log("Grid add level=%d side=%s lot=%.2f distance=%.2f step=%.2f", next_level, side, lot, distance, snapshot.step)
                    self._open_market(side, lot, snapshot.atr)

    def _maybe_enter(self, snapshot: Snapshot, foreign_lots: float) -> None:
        if self.state.get("paused", False):
            self._log("Paused: %s", self.state.get("paused_reason", ""))
            return
        if snapshot.signal not in {"BUY", "SELL"}:
            return
        if snapshot.spread_points > self.config.max_spread_points:
            self._log("Entry blocked: spread %.1f > %.1f", snapshot.spread_points, self.config.max_spread_points)
            return
        if foreign_lots > self.config.max_foreign_lots:
            self._log("Entry blocked: foreign_lots %.2f > %.2f", foreign_lots, self.config.max_foreign_lots)
            return
        lot = lot_for_level(self.config.grid, 1, 0.0)
        if lot < 0.01:
            return
        self._log("ENTRY %s lot=%.2f z=%.2f rsi=%.1f regime=%s atr=%.2f", snapshot.signal, lot, snapshot.z, snapshot.rsi, snapshot.regime, snapshot.atr)
        self._open_market(snapshot.signal, lot, snapshot.atr)

    def _open_market(self, side: str, volume: float, atr: float) -> None:
        ask, bid = self._tick_prices()
        price = ask if side == "BUY" else bid
        hard_stop = self.config.grid.hard_stop_atr * atr
        tp_dist = max(self.config.grid.take_profit_atr * atr, self._point() * 30)
        if side == "BUY":
            sl = price - hard_stop
            tp = price + tp_dist
            order_type = self.mt5.ORDER_TYPE_BUY
        else:
            sl = price + hard_stop
            tp = price - tp_dist
            order_type = self.mt5.ORDER_TYPE_SELL
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": self.config.symbol,
            "volume": float(self._round_volume(volume)),
            "type": order_type,
            "price": float(self._round_price(price)),
            "sl": float(self._round_price(sl)),
            "tp": float(self._round_price(tp)),
            "deviation": int(self.config.deviation),
            "magic": int(self.config.magic),
            "comment": "tide-wave-grid-live" if self.config.live else "tide-wave-grid-dryrun",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self._select_filling_mode(),
        }
        if not self.config.live:
            self._log("DRY-RUN order: %s", request)
            return
        result = self._send_order_with_filling_fallback(request)
        self._log("order_send result: %s", self._result_to_dict(result))

    def close_all_positions(self) -> None:
        self.close_positions(self._positions())

    def close_positions(self, positions: Sequence[Position]) -> None:
        for pos in positions:
            side = "SELL" if pos.type == self.mt5.POSITION_TYPE_BUY else "BUY"
            ask, bid = self._tick_prices()
            price = bid if side == "SELL" else ask
            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": self.config.symbol,
                "position": pos.ticket,
                "volume": float(pos.volume),
                "type": self.mt5.ORDER_TYPE_SELL if side == "SELL" else self.mt5.ORDER_TYPE_BUY,
                "price": float(self._round_price(price)),
                "deviation": int(self.config.deviation),
                "magic": int(self.config.magic),
                "comment": "tide-wave-grid-close",
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": self._select_filling_mode(),
            }
            if not self.config.live:
                self._log("DRY-RUN close: %s", request)
                continue
            result = self._send_order_with_filling_fallback(request)
            self._log("close ticket=%s result=%s", pos.ticket, self._result_to_dict(result))

    def _send_order_with_filling_fallback(self, request: Dict[str, Any]) -> Any:
        requested = int(request.get("type_filling", self.mt5.ORDER_FILLING_RETURN))
        modes = [requested, self.mt5.ORDER_FILLING_RETURN, self.mt5.ORDER_FILLING_IOC, self.mt5.ORDER_FILLING_FOK]
        tried: List[int] = []
        last_result = None
        for mode in modes:
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

    def _select_filling_mode(self) -> int:
        info = self.mt5.symbol_info(self.config.symbol)
        candidates: List[int] = []
        if info is not None:
            raw = getattr(info, "filling_mode", None)
            if raw is None:
                raw = getattr(info, "trade_fill_mode", None)
            if raw is not None:
                mapping = {0: self.mt5.ORDER_FILLING_FOK, 1: self.mt5.ORDER_FILLING_IOC, 2: self.mt5.ORDER_FILLING_RETURN, 3: self.mt5.ORDER_FILLING_BOC}
                try:
                    raw_i = int(raw)
                    if raw_i in mapping:
                        candidates.append(mapping[raw_i])
                except (TypeError, ValueError):
                    pass
        for mode in (self.mt5.ORDER_FILLING_IOC, self.mt5.ORDER_FILLING_RETURN, self.mt5.ORDER_FILLING_FOK):
            if mode not in candidates:
                candidates.append(mode)
        return candidates[0]

    def _weighted_avg_position(self, positions: Sequence[Position]) -> float:
        volume = sum(pos.volume for pos in positions)
        if volume <= 0:
            return 0.0
        return sum(pos.price_open * pos.volume for pos in positions) / volume

    def _tick_prices(self) -> Tuple[float, float]:
        tick = self.mt5.symbol_info_tick(self.config.symbol)
        if tick is None:
            raise RuntimeError(f"symbol_info_tick unavailable: {self.mt5.last_error()}")
        return float(tick.ask), float(tick.bid)

    def _spread_points(self) -> float:
        ask, bid = self._tick_prices()
        return (ask - bid) / max(self._point(), 1e-9)

    def _point(self) -> float:
        if self._point_cache is None:
            info = self.mt5.symbol_info(self.config.symbol)
            self._point_cache = float(getattr(info, "point", 0.01) if info is not None else 0.01)
        return self._point_cache

    def _digits(self) -> int:
        if self._digits_cache is None:
            info = self.mt5.symbol_info(self.config.symbol)
            self._digits_cache = int(getattr(info, "digits", 2) if info is not None else 2)
        return self._digits_cache

    def _round_price(self, value: float) -> float:
        return round(float(value), self._digits())

    def _round_volume(self, value: float) -> float:
        info = self.mt5.symbol_info(self.config.symbol)
        step = float(getattr(info, "volume_step", 0.01) if info is not None else 0.01)
        min_vol = float(getattr(info, "volume_min", 0.01) if info is not None else 0.01)
        max_vol = float(getattr(info, "volume_max", 100.0) if info is not None else 100.0)
        rounded = math.floor(float(value) / step) * step
        return round(max(min_vol, min(rounded, max_vol)), 2)

    def _tf_seconds(self) -> int:
        return {"M1": 60, "M5": 300, "M15": 900, "H1": 3600}.get(self.config.timeframe.upper(), 900)

    def _result_to_dict(self, result: Any) -> Any:
        if result is None:
            return None
        if hasattr(result, "_asdict"):
            return result._asdict()
        return str(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live Tide-Wave Grid strategy")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M15")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18812)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--loop-seconds", type=int, default=30)
    parser.add_argument("--lookback-bars", type=int, default=240)
    parser.add_argument("--max-spread-points", type=float, default=150.0)
    parser.add_argument("--max-daily-loss-pct", type=float, default=0.02)
    parser.add_argument("--max-total-dd-pct", type=float, default=0.04)
    parser.add_argument("--max-foreign-lots", type=float, default=3.0)
    parser.add_argument("--base-lot", type=float, default=0.03)
    parser.add_argument("--max-total-lots", type=float, default=0.20)
    parser.add_argument("--max-levels", type=int, default=3)
    parser.add_argument("--grid-step-atr", type=float, default=0.80)
    parser.add_argument("--take-profit-atr", type=float, default=0.35)
    parser.add_argument("--hard-stop-atr", type=float, default=3.8)
    parser.add_argument("--z-entry", type=float, default=2.0)
    parser.add_argument("--rsi-buy", type=float, default=42.0)
    parser.add_argument("--rsi-sell", type=float, default=58.0)
    parser.add_argument("--state-path", default=str(ROOT / "auto_quant/state/tide_wave_grid_state.json"))
    parser.add_argument("--log-file", default=str(ROOT / "auto_quant/logs/tide_wave_grid.log"))
    parser.add_argument("--terminal-path", default="")
    parser.add_argument("--deviation", type=int, default=30)
    parser.add_argument("--magic", type=int, default=210514)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s [%(levelname)s] %(message)s")
    grid = TideWaveGridConfig(
        name="tide_wave_grid_m15_selective_live",
        timeframe=args.timeframe,
        grid_step_atr=args.grid_step_atr,
        take_profit_atr=args.take_profit_atr,
        max_levels=args.max_levels,
        base_lot=args.base_lot,
        max_total_lots=args.max_total_lots,
        z_entry=args.z_entry,
        rsi_buy=args.rsi_buy,
        rsi_sell=args.rsi_sell,
        hard_stop_atr=args.hard_stop_atr,
        allow_trend_counter=False,
    )
    terminal_path = args.terminal_path if str(args.terminal_path).strip() else None
    config = LiveConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        host=args.host,
        port=args.port,
        live=args.live,
        loop_seconds=args.loop_seconds,
        lookback_bars=args.lookback_bars,
        max_spread_points=args.max_spread_points,
        max_daily_loss_pct=args.max_daily_loss_pct,
        max_total_dd_pct=args.max_total_dd_pct,
        max_foreign_lots=args.max_foreign_lots,
        state_path=args.state_path,
        log_file=args.log_file,
        terminal_path=terminal_path,
        deviation=args.deviation,
        magic=args.magic,
        log_level=args.log_level,
        grid=grid,
    )
    TideWaveGridLive(config).run()


if __name__ == "__main__":
    main()
