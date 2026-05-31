#!/usr/bin/env python3.14
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import logging
import math
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pymt5linux import MetaTrader5

from liquidity_sweep_strategy import LiquiditySweepLogic
from order_flow_strategy import OrderFlowLogic
from fvg_strategy import FVGLogic

LOGGER = logging.getLogger("mt5_logic_runner")
FILE_LOGGER = logging.getLogger("mt5_logic_runner.file")


@dataclasses.dataclass(frozen=True)
class RunnerConfig:
    strategy: str
    symbol: str
    timeframe: str
    host: str
    port: int
    live: bool
    risk_pct: float
    stop_atr: float
    reward_multiple: float
    long_bias: float
    trend_threshold: float
    max_lots: float
    magic: int
    deviation: int
    loop_seconds: int
    lookback_bars: int
    atr_period: int
    fast_sma: int
    slow_sma: int
    log_file: str
    terminal_path: Optional[str]


@dataclasses.dataclass
class Plan:
    action: str
    reason: str


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


@dataclasses.dataclass
class PositionState:
    ticket: int
    symbol: str
    type: int
    volume: float
    price_open: float
    sl: float
    tp: float
    profit: float


class LogicRunner:
    def __init__(self, config: RunnerConfig):
        self.config = config
        self.mt5 = MetaTrader5(host=config.host, port=config.port)
        self.symbol = config.symbol
        self.timeframe = self._resolve_timeframe(config.timeframe)
        self._last_bar_time: Optional[dt.datetime] = None
        self._initial_equity: Optional[float] = None
        self._session_start_equity: Optional[float] = None
        self._max_equity_seen: Optional[float] = None
        self._last_signal: str = "NONE"

        plan = Plan(action="NONE", reason="init")
        logic_cls = {
            "liquidity_sweep": LiquiditySweepLogic,
            "order_flow": OrderFlowLogic,
            "fvg": FVGLogic,
        }.get(config.strategy)
        if logic_cls is None:
            raise ValueError(f"Unsupported strategy: {config.strategy}")
        self.logic = logic_cls(plan)
        self.plan = plan

    def run(self) -> None:
        self._connect()
        self._prepare_symbol()
        self._seed_equity()
        self._log_file("Strategy started: %s %s live=%s", self.config.strategy, self.symbol, self.config.live)
        LOGGER.info("Strategy started: %s %s live=%s", self.config.strategy, self.symbol, self.config.live)

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
                self._log_file("Interrupted by user")
                LOGGER.info("Interrupted by user, shutting down.")
                break
            except Exception as exc:
                self._log_file("Main loop error: %s", exc)
                LOGGER.exception("Main loop error: %s", exc)
                time.sleep(self.config.loop_seconds)

        self._shutdown()

    def _log_file(self, message: str, *args: Any) -> None:
        try:
            with open(self.config.log_file, "a", encoding="utf-8") as f:
                f.write((message % args) + "\n")
        except Exception:
            pass

    def _connect(self) -> None:
        if self.config.terminal_path:
            self._log_file("Initializing terminal path: %s", self.config.terminal_path)
            LOGGER.info("Initializing terminal path: %s", self.config.terminal_path)
            ok = self.mt5.initialize(path=self.config.terminal_path)
        else:
            self._log_file("Initializing MT5 bridge without explicit terminal path")
            LOGGER.info("Initializing MT5 bridge without explicit terminal path")
            ok = self.mt5.initialize()
        self._log_file("initialize() -> %s | last_error=%s", ok, self.mt5.last_error())
        LOGGER.info("initialize() -> %s | last_error=%s", ok, self.mt5.last_error())
        if not self._is_connected():
            raise RuntimeError(f"MT5 bridge not ready: {self.mt5.last_error()}")

    def _is_connected(self) -> bool:
        try:
            ti = self.mt5.terminal_info()
            ai = self.mt5.account_info()
            return ti is not None and ai is not None
        except Exception:
            return False

    def _shutdown(self) -> None:
        try:
            self.mt5.shutdown()
        except Exception:
            pass

    def _prepare_symbol(self) -> None:
        if not self.mt5.symbol_select(self.symbol, True):
            raise RuntimeError(f"symbol_select failed for {self.symbol}: {self.mt5.last_error()}")
        info = self.mt5.symbol_info(self.symbol)
        if info is None:
            raise RuntimeError(f"symbol_info is None for {self.symbol}")
        self._log_file(
            "Symbol ready: %s | digits=%s point=%s volume_min=%s volume_step=%s volume_max=%s",
            self.symbol,
            getattr(info, "digits", "?"),
            getattr(info, "point", "?"),
            getattr(info, "volume_min", "?"),
            getattr(info, "volume_step", "?"),
            getattr(info, "volume_max", "?"),
        )
        LOGGER.info(
            "Symbol ready: %s | digits=%s point=%s volume_min=%s volume_step=%s volume_max=%s",
            self.symbol,
            getattr(info, "digits", "?"),
            getattr(info, "point", "?"),
            getattr(info, "volume_min", "?"),
            getattr(info, "volume_step", "?"),
            getattr(info, "volume_max", "?"),
        )

    def _seed_equity(self) -> None:
        equity = self._get_equity()
        self._initial_equity = equity
        self._session_start_equity = equity
        self._max_equity_seen = equity
        self._log_file("Equity seeded: %.2f", equity)
        LOGGER.info("Equity seeded: %.2f", equity)

    def _get_equity(self) -> float:
        for attempt in range(3):
            info = self.mt5.account_info()
            if info is not None:
                equity = getattr(info, "equity", None)
                if equity is None:
                    equity = getattr(info, "balance", None)
                if equity is not None:
                    return float(equity)
            self._log_file("account_info unavailable (attempt %s/3): %s", attempt + 1, self.mt5.last_error())
            LOGGER.warning("account_info unavailable (attempt %s/3): %s", attempt + 1, self.mt5.last_error())
            time.sleep(1)
            try:
                self._shutdown()
            except Exception:
                pass
            try:
                self._connect()
            except Exception as exc:
                self._log_file("Reconnect attempt failed: %s", exc)
                LOGGER.warning("Reconnect attempt failed: %s", exc)
        raise RuntimeError(f"account_info is None after reconnect attempts: {self.mt5.last_error()}")

    def _risk_guard(self) -> None:
        equity = self._get_equity()
        if self._max_equity_seen is None or equity > self._max_equity_seen:
            self._max_equity_seen = equity
        if self._initial_equity is None:
            self._initial_equity = equity
        if self._session_start_equity is None:
            self._session_start_equity = equity

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
        if len(bars) < max(self.config.slow_sma, self.config.atr_period) + 5:
            raise RuntimeError(f"Not enough bars for signal calculation: {len(bars)}")

        last_closed = bars[-2]
        closes = [bar["close"] for bar in bars[:-1]]
        highs = [bar["high"] for bar in bars[:-1]]
        lows = [bar["low"] for bar in bars[:-1]]

        atr = self._atr(bars[:-1], self.config.atr_period)
        fast_sma = statistics.fmean(closes[-self.config.fast_sma :])
        slow_sma = statistics.fmean(closes[-self.config.slow_sma :])
        momentum = (closes[-1] - closes[-4]) / atr if atr > 0 and len(closes) >= 4 else 0.0
        momentum = self._clamp(momentum, -2.0, 2.0)
        score = self._score_signal(closes, highs, lows, atr, fast_sma, slow_sma, momentum)
        signal = self._decide_signal(score, fast_sma, slow_sma, closes[-1], momentum)
        bar_time = self._bar_time(last_closed)
        return MarketSnapshot(bar_time, float(last_closed["close"]), float(last_closed["high"]), float(last_closed["low"]), atr, fast_sma, slow_sma, momentum, score, signal)

    def _fetch_bars(self) -> List[Dict[str, float]]:
        rates = self.mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, self.config.lookback_bars)
        if rates is None:
            raise RuntimeError(f"copy_rates_from_pos returned None: {self.mt5.last_error()}")
        bars: List[Dict[str, float]] = []
        for row in list(rates):
            bars.append(self._normalize_bar(row))
        return bars

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
            tr = max(curr["high"] - curr["low"], abs(curr["high"] - prev["close"]), abs(curr["low"] - prev["close"]))
            trs.append(tr)
        atr = statistics.fmean(trs)
        return max(atr, self._point() * 5)

    def _score_signal(self, closes: Sequence[float], highs: Sequence[float], lows: Sequence[float], atr: float, fast_sma: float, slow_sma: float, momentum: float) -> float:
        last_close = closes[-1]
        recent_high = max(highs[-10:])
        recent_low = min(lows[-10:])
        trend = 0.0
        if last_close > fast_sma > slow_sma:
            trend = 0.60
        elif last_close < fast_sma < slow_sma:
            trend = -0.60
        else:
            if last_close > slow_sma:
                trend = 0.20
            elif last_close < slow_sma:
                trend = -0.20
        breakout = 0.0
        if atr > 0:
            breakout = self._clamp((last_close - recent_high) / atr, -1.0, 1.0) * 0.25
            breakout += self._clamp((recent_low - last_close) / atr, -1.0, 1.0) * -0.25
        momentum_component = self._clamp(momentum, -1.5, 1.5) * 0.35
        bias_component = (self.config.long_bias - 0.5) * 0.50
        score = trend + breakout + momentum_component + bias_component
        return self._clamp(score, -1.25, 1.25)

    def _decide_signal(self, score: float, fast_sma: float, slow_sma: float, last_close: float, momentum: float) -> str:
        if score >= self.config.trend_threshold and fast_sma >= slow_sma and momentum >= -0.5:
            return "BUY"
        if score <= -self.config.trend_threshold and fast_sma <= slow_sma and momentum <= 0.5:
            return "SELL"
        return "NONE"

    def _handle_bar(self, snapshot: MarketSnapshot) -> None:
        positions = self._positions()
        plan_action = snapshot.signal
        self.plan.action = plan_action
        self.plan.reason = f"score={snapshot.score:.2f} fast={snapshot.fast_sma:.2f} slow={snapshot.slow_sma:.2f} momentum={snapshot.momentum:.2f}"
        setattr(self.logic, "plan", self.plan)

        self._log_file(
            "Bar %s | close=%.2f atr=%.2f fast=%.2f slow=%.2f momentum=%.2f score=%.2f plan=%s positions=%d",
            snapshot.bar_time,
            snapshot.close,
            snapshot.atr,
            snapshot.fast_sma,
            snapshot.slow_sma,
            snapshot.momentum,
            snapshot.score,
            plan_action,
            len(positions),
        )
        LOGGER.info(
            "Bar %s | close=%.2f atr=%.2f fast=%.2f slow=%.2f momentum=%.2f score=%.2f plan=%s positions=%d",
            snapshot.bar_time,
            snapshot.close,
            snapshot.atr,
            snapshot.fast_sma,
            snapshot.slow_sma,
            snapshot.momentum,
            snapshot.score,
            plan_action,
            len(positions),
        )

        logic_action, msg = self.logic.process_tick(snapshot.close, snapshot.bar_time.time())
        if logic_action:
            self._log_file("logic signal=%s msg=%s", logic_action, msg)
            LOGGER.info("logic signal=%s msg=%s", logic_action, msg)

        if not positions:
            if logic_action in {"BUY", "SELL"}:
                self._enter(logic_action, snapshot, msg)
            return

        pos = positions[0]
        current_side = "BUY" if pos.type == self.mt5.POSITION_TYPE_BUY else "SELL"
        if logic_action in {"BUY", "SELL"} and logic_action != current_side:
            self._log_file("Reverse signal detected; closing current position")
            LOGGER.info("Reverse signal detected; closing current position")
            self.close_all_positions()
            return
        self._maybe_trail(snapshot, pos)

    def _positions(self) -> List[PositionState]:
        raw = self.mt5.positions_get(symbol=self.symbol)
        if not raw:
            return []
        out: List[PositionState] = []
        for p in list(raw):
            out.append(PositionState(int(getattr(p, "ticket")), str(getattr(p, "symbol")), int(getattr(p, "type")), float(getattr(p, "volume")), float(getattr(p, "price_open")), float(getattr(p, "sl")), float(getattr(p, "tp")), float(getattr(p, "profit"))))
        return out

    def _maybe_trail(self, snapshot: MarketSnapshot, pos: PositionState) -> None:
        if snapshot.atr <= 0:
            return
        ask, bid = self._tick_prices()
        current_price = bid if pos.type == self.mt5.POSITION_TYPE_BUY else ask
        profit_move = (current_price - pos.price_open) if pos.type == self.mt5.POSITION_TYPE_BUY else (pos.price_open - current_price)
        if profit_move < snapshot.atr * 2.0:
            return
        if pos.type == self.mt5.POSITION_TYPE_BUY:
            new_sl = max(pos.sl, pos.price_open + snapshot.atr * 0.5)
            if pos.sl <= 0 or new_sl > pos.sl:
                self._modify_position(pos, sl=new_sl, tp=pos.tp)
        else:
            new_sl = min(pos.sl if pos.sl > 0 else current_price + snapshot.atr * 100, pos.price_open - snapshot.atr * 0.5)
            if pos.sl <= 0 or new_sl < pos.sl:
                self._modify_position(pos, sl=new_sl, tp=pos.tp)

    def _enter(self, direction: str, snapshot: MarketSnapshot, reason: str) -> None:
        ask, bid = self._tick_prices()
        price = ask if direction == "BUY" else bid
        sl, tp = self._build_sl_tp(direction, price, snapshot.atr)
        volume = self._size_position(direction, price, sl)
        if volume <= 0:
            raise RuntimeError("Calculated volume is zero")
        request = self._order_request(direction, volume, price, sl, tp)
        self._log_file("ENTRY %s volume=%.2f price=%.2f sl=%.2f tp=%.2f reason=%s", direction, volume, price, sl, tp, reason)
        LOGGER.warning("ENTRY %s volume=%.2f price=%.2f sl=%.2f tp=%.2f reason=%s", direction, volume, price, sl, tp, reason)
        if self.config.live:
            result = self._send_order_with_filling_fallback(request)
            result_dict = self._result_to_dict(result)
            self._log_file("order_send result: %s", result_dict)
            LOGGER.warning("order_send result: %s", result_dict)
            if result is None:
                self._log_file("order_send failed: %s", self.mt5.last_error())
                raise RuntimeError(f"order_send failed: {self.mt5.last_error()}")
            self._log_file("order_send confirmed: %s", result_dict)
        else:
            self._log_file("DRY-RUN request: %s", request)
            LOGGER.info("DRY-RUN request: %s", request)
        self._last_signal = direction

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
        raw_volume = self._clamp(raw_volume, float(getattr(info, "volume_min", 0.01)), float(getattr(info, "volume_max", raw_volume)))
        step = float(getattr(info, "volume_step", 0.01))
        volume = self._round_volume(raw_volume, step)
        volume = min(volume, self.config.max_lots)
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
            "comment": f"{self.config.strategy}-live",
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
                mapping = {0: self.mt5.ORDER_FILLING_FOK, 1: self.mt5.ORDER_FILLING_IOC, 2: self.mt5.ORDER_FILLING_RETURN, 3: self.mt5.ORDER_FILLING_BOC}
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
        fallback_modes = [requested, self.mt5.ORDER_FILLING_IOC, self.mt5.ORDER_FILLING_RETURN, self.mt5.ORDER_FILLING_FOK]
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
                if code not in {self.mt5.TRADE_RETCODE_INVALID_FILL, self.mt5.TRADE_RETCODE_INVALID_ORDER}:
                    return result
        return last_result

    def close_all_positions(self) -> None:
        positions = self._positions()
        if not positions:
            self._log_file("No positions to close.")
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
                self._log_file("Close result: %s", self._result_to_dict(result))
            else:
                LOGGER.info("DRY-RUN close request: %s", request)
                self._log_file("DRY-RUN close request: %s", request)

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
    parser = argparse.ArgumentParser(description="MT5 logic runner (liquidity/orderflow/fvg)")
    parser.add_argument("--strategy", choices=["liquidity_sweep", "order_flow", "fvg"], required=True)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M5")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18812)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--risk-pct", type=float, default=0.0075)
    parser.add_argument("--stop-atr", type=float, default=5.0)
    parser.add_argument("--reward-multiple", type=float, default=10.0)
    parser.add_argument("--long-bias", type=float, default=0.85)
    parser.add_argument("--trend-threshold", type=float, default=0.25)
    parser.add_argument("--max-lots", type=float, default=3.0)
    parser.add_argument("--magic", type=int, default=203493)
    parser.add_argument("--deviation", type=int, default=30)
    parser.add_argument("--loop-seconds", type=int, default=10)
    parser.add_argument("--lookback-bars", type=int, default=120)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--fast-sma", type=int, default=10)
    parser.add_argument("--slow-sma", type=int, default=30)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--terminal-path", default=r"C:\Program Files\MetaTrader 5\terminal64.exe")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = RunnerConfig(
        strategy=args.strategy,
        symbol=args.symbol,
        timeframe=args.timeframe,
        host=args.host,
        port=args.port,
        live=bool(args.live),
        risk_pct=float(args.risk_pct),
        stop_atr=float(args.stop_atr),
        reward_multiple=float(args.reward_multiple),
        long_bias=float(args.long_bias),
        trend_threshold=float(args.trend_threshold),
        max_lots=float(args.max_lots),
        magic=int(args.magic),
        deviation=int(args.deviation),
        loop_seconds=int(args.loop_seconds),
        lookback_bars=int(args.lookback_bars),
        atr_period=int(args.atr_period),
        fast_sma=int(args.fast_sma),
        slow_sma=int(args.slow_sma),
        log_file=args.log_file,
        terminal_path=args.terminal_path.strip() or None,
    )
    LogicRunner(config).run()


if __name__ == "__main__":
    main()
