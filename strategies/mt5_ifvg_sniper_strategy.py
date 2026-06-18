#!/usr/bin/env python3
"""MT5 Python port of IFVG Sniper Entry Engine [trade_w_samet].

Research/live-ready skeleton for the Python MT5/Wine bridge.
Default is dry-run. Pass --live only after explicit approval.

Core logic ported from the TradingView Pine strategy:
- Raw bullish/bearish FVG memory.
- Inversion detection with ATR/body/range quality filters.
- One active trade at a time.
- ATR stop and fixed-RR take profit.
- Conservative SL-before-TP assumption for internal status/backtest-like self-test.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pymt5linux import MetaTrader5

LOGGER = logging.getLogger("mt5_ifvg_sniper")
STATE_PATH_DEFAULT = "/home/chain4655/Documents/Sample/Python/mt5_ifvg_sniper_state.json"
LOG_PATH_DEFAULT = "/home/chain4655/Documents/Sample/Python/mt5_ifvg_sniper_strategy.log"


@dataclasses.dataclass(frozen=True)
class StrategyConfig:
    symbol: str
    timeframe: str
    host: str
    port: int
    live: bool
    loop_seconds: int
    lookback_bars: int
    max_hidden_fvg: int
    max_fvg_age: int
    min_gap_ticks: int
    filter_mode: str
    custom_min_gap_atr: float
    custom_min_body_ratio: float
    custom_min_range_atr: float
    custom_break_atr: float
    line_price_mode: str
    entry_mode: str
    order_fill_mode: str
    atr_period: int
    sl_atr_mult: float
    reward_multiple: float
    use_risk_sizing: bool
    risk_pct: float
    fixed_lots: float
    max_lots: float
    max_lots_per_order: float
    max_spread_points: float
    max_trades_per_day: int
    daily_dd_limit: float
    total_dd_limit: float
    one_position_only: bool
    allow_foreign_positions: bool
    state_path: str
    log_file: str
    terminal_path: Optional[str]
    deviation: int
    magic: int
    order_comment: str
    log_level: str


@dataclasses.dataclass
class StrategyState:
    current_day: Optional[str] = None
    day_start_equity: Optional[float] = None
    initial_equity: Optional[float] = None
    max_equity_seen: Optional[float] = None
    trades_today: int = 0
    paused: bool = False
    paused_reason: str = ""
    last_signal_time: Optional[str] = None
    last_signal_dir: int = 0
    last_entry: float = 0.0
    last_sl: float = 0.0
    last_tp: float = 0.0


@dataclasses.dataclass(frozen=True)
class IFVGSignal:
    index: int
    time: dt.datetime
    direction: int
    top: float
    bottom: float
    confirm_close: float
    line_price: float
    gap_atr: float
    body_ratio: float
    range_atr: float


class IFVGEngine:
    def __init__(self, config: StrategyConfig, point: float) -> None:
        self.config = config
        self.point = point

    def evaluate_latest(self, bars: Sequence[Dict[str, Any]]) -> Tuple[Optional[IFVGSignal], Dict[str, int]]:
        if len(bars) < self.config.atr_period + 5:
            return None, {"filtered": 0, "raw_memory": 0}

        atr_values = self._atr_values(bars, self.config.atr_period)
        raw_tops: List[float] = []
        raw_bots: List[float] = []
        raw_dirs: List[int] = []
        raw_ages: List[int] = []
        raw_gap_atr: List[float] = []
        raw_body_ratio: List[float] = []
        raw_range_atr: List[float] = []
        latest: Optional[IFVGSignal] = None
        filtered = 0

        for idx in range(2, len(bars)):
            bar = bars[idx]
            safe_atr = self._safe_atr(atr_values[idx])

            for pos in range(len(raw_ages)):
                raw_ages[pos] += 1

            for pos in range(len(raw_ages) - 1, -1, -1):
                if raw_ages[pos] > self.config.max_fvg_age:
                    self._remove_raw(pos, raw_tops, raw_bots, raw_dirs, raw_ages, raw_gap_atr, raw_body_ratio, raw_range_atr)

            min_gap = self.config.min_gap_ticks * self.point
            candle_range = max(bar["high"] - bar["low"], self.point)
            candle_body = abs(bar["close"] - bar["open"])
            body_ratio = candle_body / candle_range
            range_atr_ratio = candle_range / safe_atr

            two_back = bars[idx - 2]
            raw_bull_fvg = bar["low"] > two_back["high"] and (bar["low"] - two_back["high"]) >= min_gap
            raw_bear_fvg = bar["high"] < two_back["low"] and (two_back["low"] - bar["high"]) >= min_gap

            if raw_bull_fvg:
                raw_tops.append(bar["low"])
                raw_bots.append(two_back["high"])
                raw_dirs.append(1)
                raw_ages.append(0)
                raw_gap_atr.append((bar["low"] - two_back["high"]) / safe_atr)
                raw_body_ratio.append(body_ratio)
                raw_range_atr.append(range_atr_ratio)

            if raw_bear_fvg:
                raw_tops.append(two_back["low"])
                raw_bots.append(bar["high"])
                raw_dirs.append(-1)
                raw_ages.append(0)
                raw_gap_atr.append((two_back["low"] - bar["high"]) / safe_atr)
                raw_body_ratio.append(body_ratio)
                raw_range_atr.append(range_atr_ratio)

            while len(raw_tops) > self.config.max_hidden_fvg:
                self._remove_raw(0, raw_tops, raw_bots, raw_dirs, raw_ages, raw_gap_atr, raw_body_ratio, raw_range_atr)

            for pos in range(len(raw_tops) - 1, -1, -1):
                top = raw_tops[pos]
                bot = raw_bots[pos]
                raw_dir = raw_dirs[pos]
                clean_break_buffer = safe_atr * self._break_buffer_atr()
                bullish_inversion = raw_dir == -1 and bar["close"] > top + clean_break_buffer
                bearish_inversion = raw_dir == 1 and bar["close"] < bot - clean_break_buffer

                if bullish_inversion or bearish_inversion:
                    if self._quality_pass(raw_gap_atr[pos], raw_body_ratio[pos], raw_range_atr[pos]):
                        direction = 1 if bullish_inversion else -1
                        line_price = self._line_price(top, bot, bar["close"], direction)
                        latest = IFVGSignal(
                            index=idx,
                            time=bar["time"],
                            direction=direction,
                            top=top,
                            bottom=bot,
                            confirm_close=bar["close"],
                            line_price=line_price,
                            gap_atr=raw_gap_atr[pos],
                            body_ratio=raw_body_ratio[pos],
                            range_atr=raw_range_atr[pos],
                        )
                    else:
                        filtered += 1
                    self._remove_raw(pos, raw_tops, raw_bots, raw_dirs, raw_ages, raw_gap_atr, raw_body_ratio, raw_range_atr)
                    break

        return latest, {"filtered": filtered, "raw_memory": len(raw_tops)}

    def entry_price(self, signal: IFVGSignal, close_price: float) -> float:
        if self.config.order_fill_mode == "market_close":
            return close_price
        if self.config.entry_mode == "confirmation_close":
            return signal.confirm_close
        return signal.line_price

    def stop_take_profit(self, signal: IFVGSignal, entry: float, atr: float) -> Tuple[float, float]:
        risk = max(atr, self.point) * self.config.sl_atr_mult
        if signal.direction == 1:
            return entry - risk, entry + risk * self.config.reward_multiple
        return entry + risk, entry - risk * self.config.reward_multiple

    def _atr_values(self, bars: Sequence[Dict[str, Any]], period: int) -> List[Optional[float]]:
        trs: List[float] = []
        for idx, bar in enumerate(bars):
            if idx == 0:
                tr = bar["high"] - bar["low"]
            else:
                prev_close = bars[idx - 1]["close"]
                tr = max(bar["high"] - bar["low"], abs(bar["high"] - prev_close), abs(bar["low"] - prev_close))
            trs.append(tr)

        atrs: List[Optional[float]] = [None] * len(bars)
        if len(trs) < period:
            return atrs
        seed = sum(trs[:period]) / period
        atrs[period - 1] = seed
        prev = seed
        for idx in range(period, len(trs)):
            prev = (prev * (period - 1) + trs[idx]) / period
            atrs[idx] = prev
        return atrs

    def _safe_atr(self, value: Optional[float]) -> float:
        if value is None or value <= 0.0 or math.isnan(value):
            return self.point
        return value

    def _line_price(self, top: float, bot: float, confirm_close: float, direction: int) -> float:
        if self.config.line_price_mode == "confirmation_close":
            return confirm_close
        if self.config.line_price_mode == "midpoint":
            return (top + bot) / 2.0
        return top if direction == 1 else bot

    def _min_gap_atr(self) -> float:
        return {
            "off": 0.0,
            "loose": 0.15,
            "balanced": 0.25,
            "strict": 0.40,
        }.get(self.config.filter_mode, self.config.custom_min_gap_atr)

    def _min_body_ratio(self) -> float:
        return {
            "off": 0.0,
            "loose": 0.40,
            "balanced": 0.50,
            "strict": 0.60,
        }.get(self.config.filter_mode, self.config.custom_min_body_ratio)

    def _min_range_atr(self) -> float:
        return {
            "off": 0.0,
            "loose": 0.40,
            "balanced": 0.60,
            "strict": 0.85,
        }.get(self.config.filter_mode, self.config.custom_min_range_atr)

    def _break_buffer_atr(self) -> float:
        return {
            "off": 0.0,
            "loose": 0.0,
            "balanced": 0.05,
            "strict": 0.10,
        }.get(self.config.filter_mode, self.config.custom_break_atr)

    def _quality_pass(self, gap_atr: float, body_ratio: float, range_atr: float) -> bool:
        return self.config.filter_mode == "off" or (
            gap_atr >= self._min_gap_atr()
            and body_ratio >= self._min_body_ratio()
            and range_atr >= self._min_range_atr()
        )

    def _remove_raw(
        self,
        pos: int,
        tops: List[float],
        bots: List[float],
        dirs: List[int],
        ages: List[int],
        gap_atr: List[float],
        body_ratio: List[float],
        range_atr: List[float],
    ) -> None:
        del tops[pos]
        del bots[pos]
        del dirs[pos]
        del ages[pos]
        del gap_atr[pos]
        del body_ratio[pos]
        del range_atr[pos]


class IFVGSniperStrategy:
    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self.mt5 = MetaTrader5(host=config.host, port=config.port)
        self.symbol = config.symbol
        self.timeframe = self._resolve_timeframe(config.timeframe)
        self.state_path = Path(config.state_path)
        self.log_file = Path(config.log_file)
        self.state = self._load_state()
        self._point = 0.01
        self._digits = 2
        self._volume_min = 0.01
        self._volume_max = 100.0
        self._volume_step = 0.01
        self._contract_size = 100.0
        self._initialized = False

    def run(self) -> None:
        self._connect()
        self._prepare_symbol()
        try:
            while True:
                try:
                    self._loop_once()
                    if not self.config.live:
                        break
                    time.sleep(self.config.loop_seconds)
                except KeyboardInterrupt:
                    self._log("Interrupted by user.")
                    break
                except Exception as exc:
                    self._log("Main loop error: %s", exc)
                    if not self.config.live:
                        raise
                    self._reconnect_after_error()
        finally:
            self._shutdown()

    def _loop_once(self) -> None:
        bars = self._fetch_bars(self.config.lookback_bars)
        if len(bars) < self.config.atr_period + 5:
            raise RuntimeError(f"not enough bars: {len(bars)}")

        closed_bars = bars[:-1]
        if not closed_bars:
            return
        latest_bar = closed_bars[-1]
        self._roll_day(latest_bar["time"])
        self._risk_guard()

        engine = IFVGEngine(self.config, self._point)
        signal, stats = engine.evaluate_latest(closed_bars)
        if signal is None:
            self._log(
                "Bar %s signal=NONE close=%.2f raw_memory=%s filtered=%s paused=%s reason=%s",
                latest_bar["time"].isoformat(),
                latest_bar["close"],
                stats["raw_memory"],
                stats["filtered"],
                self.state.paused,
                self.state.paused_reason,
            )
            self.state.last_signal_time = latest_bar["time"].isoformat()
            self._save_state()
            return

        atr = engine._safe_atr(engine._atr_values(closed_bars, self.config.atr_period)[signal.index])
        entry = engine.entry_price(signal, latest_bar["close"])
        sl, tp = engine.stop_take_profit(signal, entry, atr)
        side = "BUY" if signal.direction == 1 else "SELL"
        duplicate = self.state.last_signal_time == signal.time.isoformat() and self.state.last_signal_dir == signal.direction

        self._log(
            "Bar %s signal=%s entry=%.2f sl=%.2f tp=%.2f top=%.2f bot=%.2f gap_atr=%.3f body=%.3f range_atr=%.3f duplicate=%s",
            latest_bar["time"].isoformat(),
            side,
            entry,
            sl,
            tp,
            signal.top,
            signal.bottom,
            signal.gap_atr,
            signal.body_ratio,
            signal.range_atr,
            duplicate,
        )

        if duplicate:
            return
        if self.state.paused:
            self._log("Entry blocked: paused reason=%s", self.state.paused_reason)
            return
        if self.state.trades_today >= self.config.max_trades_per_day:
            self._log("Entry blocked: trades_today=%s limit=%s", self.state.trades_today, self.config.max_trades_per_day)
            return
        if self._spread_points() > self.config.max_spread_points:
            self._log("Entry blocked: spread too wide %.1f > %.1f", self._spread_points(), self.config.max_spread_points)
            return
        if self.config.one_position_only and self._has_blocking_position():
            self._log("Entry blocked: existing position detected.")
            return

        self._enter(signal.direction, entry, sl, tp)
        self.state.last_signal_time = signal.time.isoformat()
        self.state.last_signal_dir = signal.direction
        self.state.last_entry = entry
        self.state.last_sl = sl
        self.state.last_tp = tp
        self._save_state()

    def _enter(self, direction: int, entry_basis: float, sl: float, tp: float) -> None:
        tick = self.mt5.symbol_info_tick(self.symbol)
        if tick is None:
            tick, error = self._tick_once()
            if tick is None:
                raise RuntimeError(f"symbol_info_tick failed: {error}")

        side = "BUY" if direction == 1 else "SELL"
        market_price = float(tick.ask) if direction == 1 else float(tick.bid)
        order_type = self.mt5.ORDER_TYPE_BUY if direction == 1 else self.mt5.ORDER_TYPE_SELL
        price = market_price
        action = self.mt5.TRADE_ACTION_DEAL

        if self.config.order_fill_mode == "limit_at_entry_price":
            action = self.mt5.TRADE_ACTION_PENDING
            price = entry_basis
            if direction == 1:
                order_type = self.mt5.ORDER_TYPE_BUY_LIMIT if price <= market_price else self.mt5.ORDER_TYPE_BUY_STOP
            else:
                order_type = self.mt5.ORDER_TYPE_SELL_LIMIT if price >= market_price else self.mt5.ORDER_TYPE_SELL_STOP

        volume = self._size_position(order_type, entry_basis, sl)
        if volume <= 0.0:
            self._log("Calculated volume is zero; skip entry.")
            return

        request = {
            "action": action,
            "symbol": self.symbol,
            "volume": volume,
            "type": order_type,
            "price": round(price, self._digits),
            "sl": round(sl, self._digits),
            "tp": round(tp, self._digits),
            "deviation": int(self.config.deviation),
            "magic": int(self.config.magic),
            "comment": self.config.order_comment[:31],
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self._select_filling_mode(),
        }
        self._log("ENTRY %s volume=%.2f price=%.2f sl=%.2f tp=%.2f live=%s", side, volume, price, sl, tp, self.config.live)
        if self.config.live:
            result = self._send_order_with_filling_fallback(request)
            self._log("order_send result: %s", self._result_to_dict(result))
        else:
            self._log("DRY-RUN request: %s", request)
        self.state.trades_today += 1
        self._save_state()

    def _connect(self) -> None:
        kwargs: Dict[str, Any] = {}
        if self.config.terminal_path:
            kwargs["path"] = self.config.terminal_path
        ok = self.mt5.initialize(**kwargs)
        if not ok:
            raise RuntimeError(f"MT5 initialize failed: {self.mt5.last_error()}")
        self._initialized = True
        self._log("Connected to MT5 bridge host=%s port=%s live=%s", self.config.host, self.config.port, self.config.live)

    def _prepare_symbol(self) -> None:
        if not self.mt5.symbol_select(self.symbol, True):
            self._log("symbol_select failed once, reconnecting MT5 client: %s", self.mt5.last_error())
            self._recreate_client()
            if not self.mt5.symbol_select(self.symbol, True):
                raise RuntimeError(f"symbol_select({self.symbol}) failed: {self.mt5.last_error()}")
        info = self.mt5.symbol_info(self.symbol)
        if info is None:
            self._log("symbol_info failed once, reconnecting MT5 client: %s", self.mt5.last_error())
            self._recreate_client()
            self.mt5.symbol_select(self.symbol, True)
            info = self.mt5.symbol_info(self.symbol)
        if info is None:
            raise RuntimeError(f"symbol_info({self.symbol}) failed: {self.mt5.last_error()}")
        self._point = float(getattr(info, "point", self._point) or self._point)
        self._digits = int(getattr(info, "digits", self._digits) or self._digits)
        self._volume_min = float(getattr(info, "volume_min", self._volume_min) or self._volume_min)
        self._volume_max = float(getattr(info, "volume_max", self._volume_max) or self._volume_max)
        self._volume_step = float(getattr(info, "volume_step", self._volume_step) or self._volume_step)
        self._contract_size = float(getattr(info, "trade_contract_size", self._contract_size) or self._contract_size)
        self._log(
            "Symbol ready %s point=%s digits=%s volume=[%s,%s] step=%s contract=%s",
            self.symbol,
            self._point,
            self._digits,
            self._volume_min,
            self._volume_max,
            self._volume_step,
            self._contract_size,
        )

    def _copy_rates_once(self, bars: int) -> Tuple[Optional[Any], Any]:
        client = MetaTrader5(host=self.config.host, port=self.config.port)
        try:
            ok = client.initialize()
            if not ok:
                return None, client.last_error()
            client.symbol_select(self.symbol, True)
            return client.copy_rates_from_pos(self.symbol, self.timeframe, 0, bars), client.last_error()
        except RuntimeError as exc:
            return None, exc
        finally:
            try:
                client.shutdown()
            except RuntimeError:
                pass

    def _fetch_bars(self, bars: int) -> List[Dict[str, Any]]:
        rates, error = self._copy_rates_once(bars)
        if rates is None:
            raise RuntimeError(f"copy_rates_from_pos failed: {error}")
        normalized = [self._normalize_bar(row) for row in list(rates)]
        normalized.sort(key=lambda bar: bar["time"])
        return normalized

    def _normalize_bar(self, row: Any) -> Dict[str, Any]:
        def get(name: str, index: int) -> Any:
            if hasattr(row, name):
                return getattr(row, name)
            try:
                return row[name]
            except (IndexError, KeyError, TypeError, ValueError):
                return row[index]

        timestamp = int(get("time", 0))
        return {
            "time": dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).replace(tzinfo=None),
            "open": float(get("open", 1)),
            "high": float(get("high", 2)),
            "low": float(get("low", 3)),
            "close": float(get("close", 4)),
        }

    def _has_blocking_position(self) -> bool:
        positions = self.mt5.positions_get(symbol=self.symbol)
        if positions is None:
            raise RuntimeError(f"positions_get failed: {self.mt5.last_error()}")
        if not positions:
            return False
        if not self.config.allow_foreign_positions:
            return True
        return any(int(getattr(pos, "magic", 0)) == self.config.magic for pos in positions)

    def _spread_points(self) -> float:
        tick = self.mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return 999999.0
        return abs(float(tick.ask) - float(tick.bid)) / max(self._point, 1e-12)

    def _roll_day(self, bar_time: dt.datetime) -> None:
        day = bar_time.date().isoformat()
        equity = self._equity()
        if self.state.current_day != day:
            self.state.current_day = day
            self.state.day_start_equity = equity
            self.state.trades_today = 0
            if self.state.initial_equity is None:
                self.state.initial_equity = equity
            if self.state.max_equity_seen is None:
                self.state.max_equity_seen = equity
            self._save_state()

    def _risk_guard(self) -> None:
        equity = self._equity()
        if self.state.initial_equity is None:
            self.state.initial_equity = equity
        if self.state.day_start_equity is None:
            self.state.day_start_equity = equity
        if self.state.max_equity_seen is None or equity > self.state.max_equity_seen:
            self.state.max_equity_seen = equity

        daily_dd = (self.state.day_start_equity - equity) / self.state.day_start_equity if self.state.day_start_equity else 0.0
        total_dd = (self.state.initial_equity - equity) / self.state.initial_equity if self.state.initial_equity else 0.0
        if daily_dd >= self.config.daily_dd_limit:
            self.state.paused = True
            self.state.paused_reason = f"daily_dd_hit:{daily_dd:.2%}"
        elif total_dd >= self.config.total_dd_limit:
            self.state.paused = True
            self.state.paused_reason = f"total_dd_hit:{total_dd:.2%}"
        self._save_state()

    def _equity(self) -> float:
        info = self.mt5.account_info()
        if info is None:
            return 10000.0
        equity = getattr(info, "equity", None)
        if equity is None:
            return 10000.0
        return float(equity)

    def _size_position(self, order_type: int, price: float, sl: float) -> float:
        if not self.config.use_risk_sizing:
            raw = self.config.fixed_lots
        else:
            equity = self._equity()
            risk_amount = equity * self.config.risk_pct
            risk_per_lot = self._risk_per_lot(order_type, price, sl)
            if risk_per_lot <= 0:
                return 0.0
            raw = risk_amount / risk_per_lot
        raw = min(raw, self.config.max_lots, self.config.max_lots_per_order, self._volume_max)
        raw = max(raw, self._volume_min)
        return self._round_volume(raw)

    def _risk_per_lot(self, order_type: int, price: float, sl: float) -> float:
        try:
            value = self.mt5.order_calc_profit(order_type, self.symbol, 1.0, price, sl)
            if value is not None:
                return abs(float(value))
        except RuntimeError as exc:
            self._log("order_calc_profit failed, fallback sizing: %s", exc)
        return abs(price - sl) * self._contract_size

    def _round_volume(self, volume: float) -> float:
        step = self._volume_step if self._volume_step > 0 else 0.01
        rounded = math.floor(volume / step) * step
        rounded = max(self._volume_min, min(self._volume_max, rounded))
        return round(rounded, 8)

    def _select_filling_mode(self) -> int:
        info = self.mt5.symbol_info(self.symbol)
        candidates: List[int] = []
        if info is not None:
            raw = getattr(info, "filling_mode", getattr(info, "trade_fill_mode", None))
            mapping = {0: self.mt5.ORDER_FILLING_FOK, 1: self.mt5.ORDER_FILLING_IOC, 2: self.mt5.ORDER_FILLING_RETURN}
            if raw is not None and int(raw) in mapping:
                candidates.append(mapping[int(raw)])
        for mode in (self.mt5.ORDER_FILLING_IOC, self.mt5.ORDER_FILLING_RETURN, self.mt5.ORDER_FILLING_FOK):
            if mode not in candidates:
                candidates.append(mode)
        return candidates[0]

    def _send_order_with_filling_fallback(self, request: Dict[str, Any]) -> Any:
        if request.get("action") == self.mt5.TRADE_ACTION_PENDING:
            return self.mt5.order_send(request)
        last_result = None
        tried: List[int] = []
        for mode in [request["type_filling"], self.mt5.ORDER_FILLING_RETURN, self.mt5.ORDER_FILLING_IOC, self.mt5.ORDER_FILLING_FOK]:
            if mode in tried:
                continue
            tried.append(mode)
            req = dict(request)
            req["type_filling"] = mode
            result = self.mt5.order_send(req)
            last_result = result
            self._log("order_send(type_filling=%s) -> %s", mode, self._result_to_dict(result))
            if result is None:
                continue
            code = getattr(result, "retcode", None)
            if code in {self.mt5.TRADE_RETCODE_DONE, self.mt5.TRADE_RETCODE_PLACED, self.mt5.TRADE_RETCODE_DONE_PARTIAL}:
                return result
            if code not in {self.mt5.TRADE_RETCODE_INVALID_FILL, self.mt5.TRADE_RETCODE_INVALID_ORDER}:
                return result
        return last_result

    def _result_to_dict(self, result: Any) -> Dict[str, Any]:
        if result is None:
            return {"result": None, "last_error": self.mt5.last_error()}
        if hasattr(result, "_asdict"):
            return dict(result._asdict())
        return {name: getattr(result, name) for name in dir(result) if not name.startswith("_")}

    def _tick_once(self) -> Tuple[Optional[Any], Any]:
        client = MetaTrader5(host=self.config.host, port=self.config.port)
        try:
            ok = client.initialize()
            if not ok:
                return None, client.last_error()
            client.symbol_select(self.symbol, True)
            return client.symbol_info_tick(self.symbol), client.last_error()
        except RuntimeError as exc:
            return None, exc
        finally:
            try:
                client.shutdown()
            except RuntimeError:
                pass

    def _recreate_client(self) -> None:
        try:
            self.mt5.shutdown()
        except RuntimeError:
            pass
        self.mt5 = MetaTrader5(host=self.config.host, port=self.config.port)
        self._initialized = False
        self._connect()

    def _reconnect_after_error(self) -> None:
        self._recreate_client()
        time.sleep(max(1, self.config.loop_seconds))
        self._prepare_symbol()

    def _shutdown(self) -> None:
        if self._initialized:
            try:
                self.mt5.shutdown()
            except RuntimeError:
                pass
            self._initialized = False

    def _resolve_timeframe(self, name: str) -> int:
        key = name.upper()
        mapping = {
            "M1": self.mt5.TIMEFRAME_M1,
            "M2": self.mt5.TIMEFRAME_M2,
            "M3": self.mt5.TIMEFRAME_M3,
            "M4": self.mt5.TIMEFRAME_M4,
            "M5": self.mt5.TIMEFRAME_M5,
            "M6": self.mt5.TIMEFRAME_M6,
            "M10": self.mt5.TIMEFRAME_M10,
            "M12": self.mt5.TIMEFRAME_M12,
            "M15": self.mt5.TIMEFRAME_M15,
            "M20": self.mt5.TIMEFRAME_M20,
            "M30": self.mt5.TIMEFRAME_M30,
            "H1": self.mt5.TIMEFRAME_H1,
            "H2": self.mt5.TIMEFRAME_H2,
            "H3": self.mt5.TIMEFRAME_H3,
            "H4": self.mt5.TIMEFRAME_H4,
            "D1": self.mt5.TIMEFRAME_D1,
        }
        if key not in mapping:
            raise ValueError(f"Unsupported timeframe: {name}")
        return mapping[key]

    def _load_state(self) -> StrategyState:
        if not self.state_path.exists():
            return StrategyState()
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return StrategyState(**{k: v for k, v in data.items() if k in StrategyState.__dataclass_fields__})
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            LOGGER.error("Failed to load state %s: %s", self.state_path, exc)
            return StrategyState()

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(dataclasses.asdict(self.state), indent=2, sort_keys=True), encoding="utf-8")

    def _log(self, message: str, *args: Any) -> None:
        LOGGER.info(message, *args)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MT5 IFVG Sniper Entry Engine strategy")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M5")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=18812)
    parser.add_argument("--live", action="store_true", help="Actually send orders. Default is dry-run.")
    parser.add_argument("--loop-seconds", type=int, default=10)
    parser.add_argument("--lookback-bars", type=int, default=600)
    parser.add_argument("--max-hidden-fvg", type=int, default=120)
    parser.add_argument("--max-fvg-age", type=int, default=60)
    parser.add_argument("--min-gap-ticks", type=int, default=0)
    parser.add_argument("--filter-mode", choices=["off", "loose", "balanced", "strict", "custom"], default="balanced")
    parser.add_argument("--custom-min-gap-atr", type=float, default=0.25)
    parser.add_argument("--custom-min-body-ratio", type=float, default=0.50)
    parser.add_argument("--custom-min-range-atr", type=float, default=0.60)
    parser.add_argument("--custom-break-atr", type=float, default=0.05)
    parser.add_argument("--line-price-mode", choices=["broken_boundary", "confirmation_close", "midpoint"], default="broken_boundary")
    parser.add_argument("--entry-mode", choices=["ifvg_line", "confirmation_close"], default="ifvg_line")
    parser.add_argument("--order-fill-mode", choices=["virtual_ifvg_price", "market_close", "limit_at_entry_price"], default="virtual_ifvg_price")
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--sl-atr-mult", type=float, default=1.5)
    parser.add_argument("--reward-multiple", type=float, default=3.0)
    parser.add_argument("--use-risk-sizing", action="store_true")
    parser.add_argument("--risk-pct", type=float, default=0.01, help="0.01 means 1 percent of equity")
    parser.add_argument("--fixed-lots", type=float, default=0.01)
    parser.add_argument("--max-lots", type=float, default=1.0)
    parser.add_argument("--max-lots-per-order", type=float, default=1.0)
    parser.add_argument("--max-spread-points", type=float, default=80.0)
    parser.add_argument("--max-trades-per-day", type=int, default=5)
    parser.add_argument("--daily-dd-limit", type=float, default=0.03)
    parser.add_argument("--total-dd-limit", type=float, default=0.06)
    parser.add_argument("--allow-multiple-positions", action="store_true")
    parser.add_argument("--allow-foreign-positions", action="store_true")
    parser.add_argument("--state-path", default=STATE_PATH_DEFAULT)
    parser.add_argument("--log-file", default=LOG_PATH_DEFAULT)
    parser.add_argument("--terminal-path")
    parser.add_argument("--deviation", type=int, default=30)
    parser.add_argument("--magic", type=int, default=26062031)
    parser.add_argument("--order-comment", default="IFVG_SNIPER")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--self-test", action="store_true", help="Run deterministic IFVG engine smoke test without MT5.")
    return parser


def config_from_args(args: argparse.Namespace) -> StrategyConfig:
    return StrategyConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        host=args.host,
        port=args.port,
        live=args.live,
        loop_seconds=args.loop_seconds,
        lookback_bars=args.lookback_bars,
        max_hidden_fvg=args.max_hidden_fvg,
        max_fvg_age=args.max_fvg_age,
        min_gap_ticks=args.min_gap_ticks,
        filter_mode=args.filter_mode,
        custom_min_gap_atr=args.custom_min_gap_atr,
        custom_min_body_ratio=args.custom_min_body_ratio,
        custom_min_range_atr=args.custom_min_range_atr,
        custom_break_atr=args.custom_break_atr,
        line_price_mode=args.line_price_mode,
        entry_mode=args.entry_mode,
        order_fill_mode=args.order_fill_mode,
        atr_period=args.atr_period,
        sl_atr_mult=args.sl_atr_mult,
        reward_multiple=args.reward_multiple,
        use_risk_sizing=args.use_risk_sizing,
        risk_pct=args.risk_pct,
        fixed_lots=args.fixed_lots,
        max_lots=args.max_lots,
        max_lots_per_order=args.max_lots_per_order,
        max_spread_points=args.max_spread_points,
        max_trades_per_day=args.max_trades_per_day,
        daily_dd_limit=args.daily_dd_limit,
        total_dd_limit=args.total_dd_limit,
        one_position_only=not args.allow_multiple_positions,
        allow_foreign_positions=args.allow_foreign_positions,
        state_path=args.state_path,
        log_file=args.log_file,
        terminal_path=args.terminal_path,
        deviation=args.deviation,
        magic=args.magic,
        order_comment=args.order_comment,
        log_level=args.log_level,
    )


def setup_logging(log_file: str, level: str) -> None:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
    )


def run_self_test() -> None:
    args = build_arg_parser().parse_args([])
    config = dataclasses.replace(config_from_args(args), filter_mode="off", atr_period=3, max_fvg_age=20)
    base = dt.datetime(2026, 1, 1, 0, 0)
    closes = [100, 101, 103, 104, 102, 99, 98, 101, 105, 108]
    bars: List[Dict[str, Any]] = []
    for idx, close in enumerate(closes):
        open_price = close - 0.5
        high = close + 1.0
        low = close - 1.0
        if idx == 2:
            low, high, close, open_price = 105, 107, 106, 105.5
        if idx == 5:
            low, high, close, open_price = 98, 100, 99, 99.5
        bars.append({"time": base + dt.timedelta(minutes=idx), "open": open_price, "high": high, "low": low, "close": close})
    signal, stats = IFVGEngine(config, point=0.01).evaluate_latest(bars)
    if signal is None:
        raise AssertionError(f"expected IFVG signal, stats={stats}")
    print(json.dumps({"ok": True, "dir": signal.direction, "time": signal.time.isoformat(), "line_price": signal.line_price, "stats": stats}, indent=2))


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    config = config_from_args(args)
    setup_logging(config.log_file, config.log_level)
    IFVGSniperStrategy(config).run()


if __name__ == "__main__":
    main()
