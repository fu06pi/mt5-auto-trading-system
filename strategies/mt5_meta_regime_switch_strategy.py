#!/usr/bin/env python3.14
"""Live Meta Regime Switch strategy for XAUUSD.

Uses the research-tested MetaRegimeSwitchStrategy from backtest_compare_strategies.py and
turns BUY/SELL bar signals into MT5 market orders with SL/TP. Infrastructure is intentionally
small and explicit so it can be supervised from auto_quant/active_plan.json.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from pymt5linux import MetaTrader5
except ImportError:
    sys.path.insert(0, "/home/chain4655/.openharness-venv/lib64/python3.14/site-packages")
    from pymt5linux import MetaTrader5

ROOT = Path("/home/chain4655/Documents/Projects/MT5")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_compare_strategies import Bar, MetaRegimeSwitchStrategy  # noqa: E402

LOGGER = logging.getLogger("mt5_meta_regime_switch")


@dataclasses.dataclass(frozen=True)
class Config:
    symbol: str
    timeframe: str
    host: str
    port: int
    live: bool
    loop_seconds: int
    lookback_bars: int
    max_lots: float
    max_foreign_lots: float
    max_drawdown_pct: float
    max_daily_loss_pct: float
    deviation: int
    magic: int
    state_path: Path
    log_file: Path
    terminal_path: Optional[str]


@dataclasses.dataclass(frozen=True)
class Position:
    ticket: int
    type: int
    volume: float
    price_open: float
    sl: float
    tp: float
    profit: float
    magic: int
    comment: str


class MetaRegimeSwitchLive:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.mt5 = MetaTrader5(host=config.host, port=config.port)
        self.strategy = MetaRegimeSwitchStrategy()
        self._last_bar_time: Optional[dt.datetime] = None
        self._initial_equity: Optional[float] = None
        self._day_start_equity: Optional[float] = None
        self._max_equity_seen: Optional[float] = None
        self._paused = False

    def run(self) -> None:
        self._setup_logging()
        self._connect()
        self._prepare_symbol()
        self._seed_equity()
        self._log("Strategy started: meta_regime_switch %s %s live=%s", self.config.symbol, self.config.timeframe, self.config.live)
        while True:
            try:
                bars = self._fetch_bars()
                if len(bars) < 240:
                    self._log("Not enough bars: %s", len(bars))
                    time.sleep(self.config.loop_seconds)
                    continue
                bar_time = bars[-1].time
                if self._last_bar_time is not None and bar_time <= self._last_bar_time:
                    time.sleep(self.config.loop_seconds)
                    continue
                self._last_bar_time = bar_time
                self._handle_bar(bars)
            except KeyboardInterrupt:
                self._log("Interrupted by user")
                break
            except Exception as exc:
                self._log("Main loop error: %s", exc)
                LOGGER.exception("Main loop error: %s", exc)
                self._reconnect_quietly()
                time.sleep(self.config.loop_seconds)
        self._shutdown()

    def _setup_logging(self) -> None:
        self.config.log_file.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(self.config.log_file, encoding="utf-8"), logging.StreamHandler()],
        )

    def _log(self, message: str, *args: Any) -> None:
        LOGGER.info(message, *args)

    def _connect(self) -> None:
        terminal_path = self.config.terminal_path
        ok = self.mt5.initialize(path=terminal_path) if terminal_path else self.mt5.initialize()
        self._log("initialize() -> %s | last_error=%s", ok, self.mt5.last_error())
        if not ok or self.mt5.account_info() is None:
            raise RuntimeError(f"MT5 bridge not ready: {self.mt5.last_error()}")

    def _shutdown(self) -> None:
        try:
            self.mt5.shutdown()
        except Exception:
            pass

    def _reconnect_quietly(self) -> None:
        try:
            self._shutdown()
            self.mt5 = MetaTrader5(host=self.config.host, port=self.config.port)
            self._connect()
            self._prepare_symbol()
        except Exception as exc:
            self._log("Reconnect failed: %s", exc)

    def _prepare_symbol(self) -> None:
        for attempt in range(3):
            if self.mt5.symbol_select(self.config.symbol, True):
                info = self.mt5.symbol_info(self.config.symbol)
                if info is not None:
                    self._log(
                        "Symbol ready: %s | digits=%s point=%s volume_min=%s volume_step=%s volume_max=%s",
                        self.config.symbol,
                        getattr(info, "digits", "?"),
                        getattr(info, "point", "?"),
                        getattr(info, "volume_min", "?"),
                        getattr(info, "volume_step", "?"),
                        getattr(info, "volume_max", "?"),
                    )
                    return
            self._log("symbol prepare retry %s/3: %s", attempt + 1, self.mt5.last_error())
            self._shutdown()
            self.mt5 = MetaTrader5(host=self.config.host, port=self.config.port)
            self._connect()
            time.sleep(1)
        raise RuntimeError(f"symbol_info is None for {self.config.symbol}")

    def _seed_equity(self) -> None:
        equity = self._equity()
        self._initial_equity = equity
        self._day_start_equity = equity
        self._max_equity_seen = equity
        self._write_state({"started_at": dt.datetime.now().isoformat(), "initial_equity": equity, "day_start_equity": equity, "max_equity_seen": equity, "paused": False})
        self._log("Equity seeded: %.2f", equity)

    def _write_state(self, payload: Dict[str, Any]) -> None:
        self.config.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
        import json

        self.config.state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _equity(self) -> float:
        account = self.mt5.account_info()
        if account is None:
            raise RuntimeError(f"account_info unavailable: {self.mt5.last_error()}")
        return float(getattr(account, "equity", getattr(account, "balance", 0.0)))

    def _risk_guard(self) -> bool:
        equity = self._equity()
        if self._initial_equity is None or self._day_start_equity is None or self._max_equity_seen is None:
            self._seed_equity()
        assert self._initial_equity is not None
        assert self._day_start_equity is not None
        assert self._max_equity_seen is not None
        self._max_equity_seen = max(self._max_equity_seen, equity)
        total_dd = (self._max_equity_seen - equity) / max(self._max_equity_seen, 1e-9)
        daily_dd = (self._day_start_equity - equity) / max(self._day_start_equity, 1e-9)
        self._paused = total_dd >= self.config.max_drawdown_pct or daily_dd >= self.config.max_daily_loss_pct
        self._write_state({"initial_equity": self._initial_equity, "day_start_equity": self._day_start_equity, "max_equity_seen": self._max_equity_seen, "last_equity": equity, "total_dd_pct": round(total_dd * 100, 3), "daily_dd_pct": round(daily_dd * 100, 3), "paused": self._paused})
        if self._paused:
            self._log("Risk paused: equity=%.2f total_dd=%.2f%% daily_dd=%.2f%%", equity, total_dd * 100, daily_dd * 100)
        return not self._paused

    def _fetch_bars(self) -> List[Bar]:
        # pymt5linux on this Wine setup can keep account_info() healthy while
        # market-data calls fail on a stale client. Recreate the client before
        # each data pull, matching the recovery pattern used by the stable live
        # strategies in this repo.
        self._shutdown()
        self.mt5 = MetaTrader5(host=self.config.host, port=self.config.port)
        self._connect()
        if not self.mt5.symbol_select(self.config.symbol, True):
            raise RuntimeError(f"symbol_select failed for {self.config.symbol}: {self.mt5.last_error()}")
        tf = {"M5": self.mt5.TIMEFRAME_M5, "M15": self.mt5.TIMEFRAME_M15, "H1": self.mt5.TIMEFRAME_H1}[self.config.timeframe.upper()]
        rates = self.mt5.copy_rates_from_pos(self.config.symbol, tf, 0, self.config.lookback_bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"copy_rates_from_pos failed: {self.mt5.last_error()}")
        bars = [Bar(time=dt.datetime.fromtimestamp(int(row["time"])), open=float(row["open"]), high=float(row["high"]), low=float(row["low"]), close=float(row["close"]), tick_volume=float(row["tick_volume"])) for row in rates]
        bars.sort(key=lambda item: item.time)
        return bars

    def _positions(self) -> List[Position]:
        raw = self.mt5.positions_get(symbol=self.config.symbol)
        if not raw:
            return []
        positions: List[Position] = []
        for pos in list(raw):
            positions.append(Position(ticket=int(getattr(pos, "ticket")), type=int(getattr(pos, "type")), volume=float(getattr(pos, "volume")), price_open=float(getattr(pos, "price_open")), sl=float(getattr(pos, "sl")), tp=float(getattr(pos, "tp")), profit=float(getattr(pos, "profit")), magic=int(getattr(pos, "magic", 0) or 0), comment=str(getattr(pos, "comment", "") or "")))
        return positions

    def _handle_bar(self, bars: Sequence[Bar]) -> None:
        signal, score, atr_value = self.strategy.signal(bars)
        selected = getattr(self.strategy, "selected_name", "none")
        positions = self._positions()
        own = [pos for pos in positions if pos.magic == self.config.magic]
        foreign_lots = sum(pos.volume for pos in positions if pos.magic != self.config.magic)
        self._log(
            "Bar %s | close=%.2f atr=%.2f signal=%s score=%.2f selected=%s positions=%d own=%d foreign_lots=%.2f",
            bars[-1].time,
            bars[-1].close,
            atr_value,
            signal,
            score,
            selected,
            len(positions),
            len(own),
            foreign_lots,
        )
        if signal not in {"BUY", "SELL"} or atr_value <= 0:
            return
        if not self._risk_guard():
            return
        if foreign_lots > self.config.max_foreign_lots:
            self._log("ENTRY_FILTERED foreign_lots %.2f > %.2f", foreign_lots, self.config.max_foreign_lots)
            return
        if own:
            own_side = "BUY" if own[0].type == self.mt5.POSITION_TYPE_BUY else "SELL"
            if own_side != signal:
                self._log("Reverse signal; closing own positions before new entry")
                self._close_positions(own)
            return
        self._enter(signal, atr_value, selected, score)

    def _tick_prices(self) -> Tuple[float, float]:
        tick = self.mt5.symbol_info_tick(self.config.symbol)
        if tick is None:
            raise RuntimeError(f"symbol_info_tick failed: {self.mt5.last_error()}")
        ask = float(getattr(tick, "ask", 0.0) or getattr(tick, "last", 0.0))
        bid = float(getattr(tick, "bid", 0.0) or getattr(tick, "last", 0.0))
        if ask <= 0 or bid <= 0:
            raise RuntimeError(f"invalid tick ask={ask} bid={bid}")
        return ask, bid

    def _enter(self, direction: str, atr_value: float, selected: str, score: float) -> None:
        ask, bid = self._tick_prices()
        price = ask if direction == "BUY" else bid
        sl, tp = self.strategy.sl_tp(direction, price, atr_value, score)
        digits = self._digits()
        sl = round(sl, digits)
        tp = round(tp, digits)
        risk_pct = min(float(getattr(self.strategy, "risk_pct", 0.005)), 0.006)
        volume = self._size_position(direction, price, sl, risk_pct)
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": self.config.symbol,
            "volume": volume,
            "type": self.mt5.ORDER_TYPE_BUY if direction == "BUY" else self.mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": self.config.deviation,
            "magic": self.config.magic,
            "comment": f"meta::{selected}",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self._select_filling_mode(),
        }
        self._log("ENTRY %s volume=%.2f price=%.2f sl=%.2f tp=%.2f selected=%s score=%.2f", direction, volume, price, sl, tp, selected, score)
        if self.config.live:
            result = self._send_order_with_filling_fallback(request)
            self._log("order_send result: %s", self._result_to_dict(result))
        else:
            self._log("DRY-RUN request: %s", request)

    def _size_position(self, direction: str, price: float, sl: float, risk_pct: float) -> float:
        equity = self._equity()
        risk_amount = equity * risk_pct
        risk_per_lot = self._risk_per_lot(direction, price, sl)
        if risk_per_lot <= 0:
            raise RuntimeError("risk_per_lot invalid")
        raw = min(risk_amount / risk_per_lot, self.config.max_lots)
        margin_lot = self._margin_per_lot(direction, price)
        if margin_lot > 0:
            account = self.mt5.account_info()
            free_margin = float(getattr(account, "margin_free", equity) if account is not None else equity)
            raw = min(raw, (free_margin * 0.85) / margin_lot)
        info = self.mt5.symbol_info(self.config.symbol)
        if info is None:
            raise RuntimeError("symbol_info unavailable")
        step = float(getattr(info, "volume_step", 0.01))
        min_vol = float(getattr(info, "volume_min", 0.01))
        max_vol = min(float(getattr(info, "volume_max", self.config.max_lots)), self.config.max_lots)
        volume = math.floor(max(min_vol, min(raw, max_vol)) / step) * step
        return round(max(min_vol, min(volume, max_vol)), 2)

    def _risk_per_lot(self, direction: str, price: float, sl: float) -> float:
        order_type = self.mt5.ORDER_TYPE_BUY if direction == "BUY" else self.mt5.ORDER_TYPE_SELL
        value = self.mt5.order_calc_profit(order_type, self.config.symbol, 1.0, price, sl)
        if value is not None:
            return abs(float(value))
        info = self.mt5.symbol_info(self.config.symbol)
        contract_size = float(getattr(info, "trade_contract_size", 100.0) if info is not None else 100.0)
        return abs(price - sl) * contract_size

    def _margin_per_lot(self, direction: str, price: float) -> float:
        order_type = self.mt5.ORDER_TYPE_BUY if direction == "BUY" else self.mt5.ORDER_TYPE_SELL
        value = self.mt5.order_calc_margin(order_type, self.config.symbol, 1.0, price)
        return abs(float(value)) if value is not None else 0.0

    def _close_positions(self, positions: Sequence[Position]) -> None:
        ask, bid = self._tick_prices()
        for pos in positions:
            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": self.config.symbol,
                "volume": pos.volume,
                "type": self.mt5.ORDER_TYPE_SELL if pos.type == self.mt5.POSITION_TYPE_BUY else self.mt5.ORDER_TYPE_BUY,
                "position": pos.ticket,
                "price": bid if pos.type == self.mt5.POSITION_TYPE_BUY else ask,
                "deviation": self.config.deviation,
                "magic": self.config.magic,
                "comment": "meta-reverse-close",
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": self._select_filling_mode(),
            }
            if self.config.live:
                result = self._send_order_with_filling_fallback(request)
                self._log("Close result ticket=%s: %s", pos.ticket, self._result_to_dict(result))
            else:
                self._log("DRY-RUN close request: %s", request)

    def _select_filling_mode(self) -> int:
        info = self.mt5.symbol_info(self.config.symbol)
        candidates: List[int] = []
        if info is not None:
            raw = getattr(info, "filling_mode", None)
            mapping = {0: self.mt5.ORDER_FILLING_FOK, 1: self.mt5.ORDER_FILLING_IOC, 2: self.mt5.ORDER_FILLING_RETURN, 3: self.mt5.ORDER_FILLING_BOC}
            if raw is not None and int(raw) in mapping:
                candidates.append(mapping[int(raw)])
        for mode in (self.mt5.ORDER_FILLING_IOC, self.mt5.ORDER_FILLING_RETURN, self.mt5.ORDER_FILLING_FOK):
            if mode not in candidates:
                candidates.append(mode)
        return candidates[0]

    def _send_order_with_filling_fallback(self, request: Dict[str, Any]) -> Any:
        modes = [int(request.get("type_filling", self.mt5.ORDER_FILLING_IOC)), self.mt5.ORDER_FILLING_IOC, self.mt5.ORDER_FILLING_RETURN, self.mt5.ORDER_FILLING_FOK]
        tried: List[int] = []
        last = None
        for mode in modes:
            if mode in tried:
                continue
            tried.append(mode)
            req = dict(request)
            req["type_filling"] = mode
            result = self.mt5.order_send(req)
            last = result
            self._log("order_send(type_filling=%s) -> %s", mode, self._result_to_dict(result))
            code = getattr(result, "retcode", None) if result is not None else None
            if code in {self.mt5.TRADE_RETCODE_DONE, self.mt5.TRADE_RETCODE_PLACED, self.mt5.TRADE_RETCODE_DONE_PARTIAL}:
                return result
            if code not in {self.mt5.TRADE_RETCODE_INVALID_FILL, self.mt5.TRADE_RETCODE_INVALID_ORDER}:
                return result
        return last

    def _digits(self) -> int:
        info = self.mt5.symbol_info(self.config.symbol)
        return int(getattr(info, "digits", 2) if info is not None else 2)

    def _result_to_dict(self, result: Any) -> Any:
        if result is None:
            return None
        if hasattr(result, "_asdict"):
            return result._asdict()
        return str(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live Meta Regime Switch strategy")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M15")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18812)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--loop-seconds", type=int, default=30)
    parser.add_argument("--lookback-bars", type=int, default=260)
    parser.add_argument("--max-lots", type=float, default=1.2)
    parser.add_argument("--max-foreign-lots", type=float, default=3.0)
    parser.add_argument("--max-drawdown-pct", type=float, default=0.05)
    parser.add_argument("--max-daily-loss-pct", type=float, default=0.018)
    parser.add_argument("--deviation", type=int, default=35)
    parser.add_argument("--magic", type=int, default=230520)
    parser.add_argument("--state-path", default=str(ROOT / "auto_quant/state/meta_regime_switch_state.json"))
    parser.add_argument("--log-file", default=str(ROOT / "auto_quant/logs/meta_regime_switch.log"))
    parser.add_argument("--terminal-path", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = Config(
        symbol=args.symbol,
        timeframe=args.timeframe,
        host=args.host,
        port=args.port,
        live=args.live,
        loop_seconds=args.loop_seconds,
        lookback_bars=args.lookback_bars,
        max_lots=args.max_lots,
        max_foreign_lots=args.max_foreign_lots,
        max_drawdown_pct=args.max_drawdown_pct,
        max_daily_loss_pct=args.max_daily_loss_pct,
        deviation=args.deviation,
        magic=args.magic,
        state_path=Path(args.state_path),
        log_file=Path(args.log_file),
        terminal_path=args.terminal_path if str(args.terminal_path).strip() else None,
    )
    MetaRegimeSwitchLive(config).run()


if __name__ == "__main__":
    main()
