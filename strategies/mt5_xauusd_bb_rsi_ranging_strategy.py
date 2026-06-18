#!/usr/bin/env python3
"""Low-risk MT5 live sleeve for XAUUSD M5 BB+RSI ranging mean reversion.

Research-backed candidate:
- BB(20, 2.0)
- RSI long <= 40 / short >= 60
- ADX <= 20
- EMA50 slope / ATR <= 0.20
- ATR14 / rolling 20-day median ATR between 0.65 and 1.65
- SL: rejection bar extreme +/- 1 ATR
- TP: opposite Bollinger band
- Early exit: close if an opposite valid BB+RSI setup appears

No partial TP, no break-even, no trailing stop by default.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import logging
import math
import statistics
import time
from typing import Any, List, Optional, Sequence, Tuple

from mt5_doomsday_strategy import (
    DoomsdayMT5Strategy,
    MarketSnapshot,
    StrategyConfig as DoomsdayStrategyConfig,
    build_parser as build_doomsday_parser,
)

LOGGER = logging.getLogger("mt5_xauusd_bb_rsi_ranging_strategy")


@dataclasses.dataclass(frozen=True)
class BbRsiConfig(DoomsdayStrategyConfig):
    bb_period: int
    bb_std: float
    rsi_period: int
    rsi_long_max: float
    rsi_short_min: float
    adx_max: float
    ema_slope_period: int
    ema_slope_lookback: int
    slope_atr_max: float
    atr_ratio_min: float
    atr_ratio_max: float
    atr_median_bars: int
    max_spread_points: float
    max_hold_bars: int
    trend_conflict_momentum_threshold: float
    trend_conflict_ema_buffer_atr: float
    once: bool = False
    self_test: bool = False


@dataclasses.dataclass
class BbRsiSnapshot(MarketSnapshot):
    bb_mid: float = 0.0
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    rsi: float = 50.0
    adx: float = 50.0
    ema50: float = 0.0
    ema_slope_atr: float = 999.0
    atr_ratio: float = 999.0
    spread_points: float = 0.0
    rejection_long: bool = False
    rejection_short: bool = False
    block_reason: str = ""


class BbRsiRangingMT5Strategy(DoomsdayMT5Strategy):
    config: BbRsiConfig

    def __init__(self, config: BbRsiConfig):
        super().__init__(config)
        self._entry_bar_time: Optional[dt.datetime] = None
        self._entry_snapshot: Optional[BbRsiSnapshot] = None

    def run_once(self) -> None:
        self._connect()
        self._prepare_symbol()
        self._seed_equity()
        self._risk_guard()
        snapshot = self._build_snapshot()
        self._handle_bar(snapshot)
        self._shutdown()

    def _build_snapshot(self) -> BbRsiSnapshot:
        need = max(
            self.config.lookback_bars,
            self.config.atr_median_bars + self.config.atr_period + 10,
            self.config.bb_period + self.config.rsi_period + self.config.ema_slope_period + 20,
        )
        bars = self._fetch_bars(min_count=need)
        if len(bars) < need:
            raise RuntimeError(f"Not enough bars for BB+RSI calculation: got={len(bars)} need={need}")

        closed = bars[:-1]
        last = closed[-1]
        closes = [float(bar["close"]) for bar in closed]
        highs = [float(bar["high"]) for bar in closed]
        lows = [float(bar["low"]) for bar in closed]

        atr = self._atr(closed, self.config.atr_period)
        bb_mid, bb_upper, bb_lower = self._bollinger(closes, self.config.bb_period, self.config.bb_std)
        rsi = self._rsi(closes, self.config.rsi_period)
        adx = self._adx(highs, lows, closes, self.config.atr_period)
        ema_series = self._ema_series(closes, self.config.ema_slope_period)
        if len(ema_series) <= self.config.ema_slope_lookback or math.isnan(ema_series[-1]):
            raise RuntimeError("EMA slope not ready")
        ema_now = ema_series[-1]
        ema_prev = ema_series[-1 - self.config.ema_slope_lookback]
        ema_slope_atr = abs(ema_now - ema_prev) / max(atr, self._point())
        atr_ratio = self._atr_ratio(closed, self.config.atr_period, self.config.atr_median_bars, atr)
        spread_points = self._spread_points()

        low = float(last["low"])
        high = float(last["high"])
        close = float(last["close"])
        rejection_long = low <= bb_lower and close > bb_lower and rsi <= self.config.rsi_long_max
        rejection_short = high >= bb_upper and close < bb_upper and rsi >= self.config.rsi_short_min
        momentum = (closes[-1] - closes[-4]) / max(atr, self._point()) if len(closes) >= 4 else 0.0

        signal = "NONE"
        block = ""
        if spread_points > self.config.max_spread_points:
            block = f"spread {spread_points:.1f}>{self.config.max_spread_points:.1f}"
        elif adx > self.config.adx_max:
            block = f"adx {adx:.1f}>{self.config.adx_max:.1f}"
        elif ema_slope_atr > self.config.slope_atr_max:
            block = f"slope_atr {ema_slope_atr:.2f}>{self.config.slope_atr_max:.2f}"
        elif not (self.config.atr_ratio_min <= atr_ratio <= self.config.atr_ratio_max):
            block = f"atr_ratio {atr_ratio:.2f} outside {self.config.atr_ratio_min:.2f}-{self.config.atr_ratio_max:.2f}"
        elif self._trend_conflict_blocks(
            direction="BUY" if rejection_long else "SELL" if rejection_short else "NONE",
            close=close,
            ema=ema_now,
            atr=atr,
            momentum=momentum,
        ):
            block = (
                f"trend_conflict dir={'BUY' if rejection_long else 'SELL'} "
                f"close={close:.2f} ema={ema_now:.2f} mom={momentum:.2f}"
            )
        elif rejection_long:
            signal = "BUY"
        elif rejection_short:
            signal = "SELL"

        score = 1.0 if signal == "BUY" else -1.0 if signal == "SELL" else 0.0
        return BbRsiSnapshot(
            bar_time=self._bar_time(last),
            close=close,
            high=high,
            low=low,
            atr=atr,
            fast_sma=bb_mid,
            slow_sma=ema_now,
            momentum=momentum,
            score=score,
            signal=signal,
            atr_pct=atr / max(abs(close), self._point()),
            range_atr=(max(highs[-20:]) - min(lows[-20:])) / max(atr, self._point()),
            spike_atr=(high - low) / max(atr, self._point()),
            breakout_atr=0.0,
            close_location=(close - low) / max(high - low, self._point()),
            high_volatility=False,
            bb_mid=bb_mid,
            bb_upper=bb_upper,
            bb_lower=bb_lower,
            rsi=rsi,
            adx=adx,
            ema50=ema_now,
            ema_slope_atr=ema_slope_atr,
            atr_ratio=atr_ratio,
            spread_points=spread_points,
            rejection_long=rejection_long,
            rejection_short=rejection_short,
            block_reason=block,
        )

    def _handle_bar(self, snapshot: MarketSnapshot) -> None:
        bb = snapshot if isinstance(snapshot, BbRsiSnapshot) else None
        positions = self._positions()
        foreign = [pos for pos in self._all_positions() if pos.symbol != self.symbol]
        if bb is None:
            return

        self._log_file(
            "Bar %s | close=%.2f atr=%.2f bb_mid=%.2f bb_upper=%.2f bb_lower=%.2f rsi=%.1f adx=%.1f slope_atr=%.2f atr_ratio=%.2f spread=%.1f rej_long=%s rej_short=%s signal=%s block=%s positions=%d foreign=%d",
            bb.bar_time,
            bb.close,
            bb.atr,
            bb.bb_mid,
            bb.bb_upper,
            bb.bb_lower,
            bb.rsi,
            bb.adx,
            bb.ema_slope_atr,
            bb.atr_ratio,
            bb.spread_points,
            bb.rejection_long,
            bb.rejection_short,
            bb.signal,
            bb.block_reason or "none",
            len(positions),
            len(foreign),
        )
        LOGGER.info(
            "Bar %s | BBRSI signal=%s rsi=%.1f adx=%.1f atr_ratio=%.2f spread=%.1f positions=%d block=%s",
            bb.bar_time,
            bb.signal,
            bb.rsi,
            bb.adx,
            bb.atr_ratio,
            bb.spread_points,
            len(positions),
            bb.block_reason or "none",
        )

        if positions:
            pos = positions[0]
            held_bars = self._held_bars(bb.bar_time)
            if self._is_opposite_signal(bb, pos):
                self._log_file("SIGNAL_FLIP_EXIT ticket=%s pos_type=%s new_signal=%s", pos.ticket, pos.type, bb.signal)
                LOGGER.warning("SIGNAL_FLIP_EXIT ticket=%s new_signal=%s", pos.ticket, bb.signal)
                self.close_all_positions()
                self._last_trade_close_time = bb.bar_time
                self._entry_bar_time = None
                self._entry_snapshot = None
                return
            if self.config.max_hold_bars > 0 and held_bars >= self.config.max_hold_bars:
                self._log_file("MAX_HOLD_EXIT ticket=%s held_bars=%d", pos.ticket, held_bars)
                LOGGER.warning("MAX_HOLD_EXIT ticket=%s held_bars=%d", pos.ticket, held_bars)
                self.close_all_positions()
                self._last_trade_close_time = bb.bar_time
                self._entry_bar_time = None
                self._entry_snapshot = None
                return
            return

        if bb.signal != "NONE" and self._cooldown_ok(bb.bar_time):
            self._entry_snapshot = bb
            self._entry_bar_time = bb.bar_time
            self._enter(bb)

    def _is_opposite_signal(self, snapshot: BbRsiSnapshot, pos: Any) -> bool:
        if pos.type == self.mt5.POSITION_TYPE_BUY and snapshot.signal == "SELL":
            return True
        if pos.type == self.mt5.POSITION_TYPE_SELL and snapshot.signal == "BUY":
            return True
        return False

    def _trend_conflict_blocks(
        self,
        direction: str,
        close: float,
        ema: float,
        atr: float,
        momentum: float,
    ) -> bool:
        threshold = float(self.config.trend_conflict_momentum_threshold)
        if threshold <= 0 or direction not in {"BUY", "SELL"}:
            return False
        buffer = float(self.config.trend_conflict_ema_buffer_atr) * max(atr, self._point())
        if direction == "BUY":
            return close <= ema - buffer and momentum <= -threshold
        return close >= ema + buffer and momentum >= threshold

    def _held_bars(self, bar_time: dt.datetime) -> int:
        if self._entry_bar_time is None:
            return 0
        minutes = max(0.0, (bar_time - self._entry_bar_time).total_seconds() / 60.0)
        tf_minutes = 5.0 if self.config.timeframe.upper() == "M5" else 1.0
        return int(minutes // tf_minutes)

    def _build_sl_tp(self, direction: str, price: float, atr: float, score: float) -> Tuple[float, float]:
        snap = self._entry_snapshot
        point = self._point()
        digits = self._digits()
        if snap is None:
            return super()._build_sl_tp(direction, price, atr, score)
        if direction == "BUY":
            sl = snap.low - atr * self.config.stop_atr
            tp = max(snap.bb_upper, price + point)
        else:
            sl = snap.high + atr * self.config.stop_atr
            tp = min(snap.bb_lower, price - point)
        if abs(price - sl) < point:
            sl = price - point if direction == "BUY" else price + point
        if abs(price - tp) < point:
            tp = price + point if direction == "BUY" else price - point
        return self._round_to_digits(sl, digits), self._round_to_digits(tp, digits)

    def _order_request(self, direction: str, volume: float, price: float, sl: float, tp: float) -> dict:
        request = super()._order_request(direction, volume, price, sl, tp)
        request["comment"] = "bbrsi-range-live" if self.config.live else "bbrsi-range-dry"
        return request

    def _spread_points(self) -> float:
        tick = self.mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return 999999.0
        return max(0.0, (float(tick.ask) - float(tick.bid)) / max(self._point(), 1e-12))

    @staticmethod
    def _bollinger(values: Sequence[float], period: int, std_mult: float) -> Tuple[float, float, float]:
        window = list(values[-period:])
        mid = statistics.fmean(window)
        std = statistics.pstdev(window) if len(window) > 1 else 0.0
        return mid, mid + std_mult * std, mid - std_mult * std

    @staticmethod
    def _ema_series(values: Sequence[float], length: int) -> List[float]:
        out: List[float] = []
        alpha = 2.0 / (length + 1.0)
        ema: Optional[float] = None
        for i, value in enumerate(values):
            if i + 1 < length:
                out.append(math.nan)
                continue
            if ema is None:
                ema = statistics.fmean(values[i + 1 - length : i + 1])
            else:
                ema = alpha * value + (1.0 - alpha) * ema
            out.append(float(ema))
        return out

    @staticmethod
    def _rsi(values: Sequence[float], period: int) -> float:
        if len(values) < period + 1:
            return 50.0
        gains: List[float] = []
        losses: List[float] = []
        for i in range(len(values) - period, len(values)):
            change = values[i] - values[i - 1]
            gains.append(max(change, 0.0))
            losses.append(max(-change, 0.0))
        avg_gain = statistics.fmean(gains)
        avg_loss = statistics.fmean(losses)
        if avg_loss <= 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _adx(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int) -> float:
        if len(closes) < period * 2 + 2:
            return 50.0
        trs: List[float] = []
        plus_dm: List[float] = []
        minus_dm: List[float] = []
        for i in range(1, len(closes)):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
            up_move = highs[i] - highs[i - 1]
            down_move = lows[i - 1] - lows[i]
            trs.append(tr)
            plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
            minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
        dxs: List[float] = []
        for i in range(period - 1, len(trs)):
            tr_sum = sum(trs[i + 1 - period : i + 1])
            if tr_sum <= 0:
                continue
            plus_di = 100.0 * sum(plus_dm[i + 1 - period : i + 1]) / tr_sum
            minus_di = 100.0 * sum(minus_dm[i + 1 - period : i + 1]) / tr_sum
            denom = plus_di + minus_di
            if denom > 0:
                dxs.append(100.0 * abs(plus_di - minus_di) / denom)
        return float(statistics.fmean(dxs[-period:])) if len(dxs) >= period else 50.0

    def _atr_ratio(self, bars: Sequence[dict], period: int, median_bars: int, current_atr: float) -> float:
        atrs: List[float] = []
        start = max(period + 1, len(bars) - median_bars)
        for end in range(start, len(bars) + 1):
            window = bars[:end]
            try:
                atrs.append(self._atr(window, period))
            except RuntimeError:
                continue
        if not atrs:
            return 999.0
        baseline = statistics.median(atrs)
        return current_atr / max(baseline, self._point())


def build_parser() -> argparse.ArgumentParser:
    parser = build_doomsday_parser()
    parser.description = "MT5 XAUUSD M5 BB+RSI ranging sleeve."
    parser.set_defaults(
        symbol="XAUUSD",
        timeframe="M5",
        risk_pct=0.0005,
        stop_atr=1.0,
        reward_multiple=1.0,
        cooldown_minutes=20,
        max_drawdown_pct=0.010,
        max_daily_loss_pct=0.006,
        max_lots=0.05,
        max_leverage=1.0,
        magic=26052035,
        deviation=30,
        loop_seconds=10,
        lookback_bars=6200,
        atr_period=14,
        fast_sma=20,
        slow_sma=50,
        high_vol_only=False,
        terminal_path="",
    )
    parser.add_argument("--bb-period", type=int, default=20)
    parser.add_argument("--bb-std", type=float, default=2.0)
    parser.add_argument("--rsi-period", type=int, default=14)
    parser.add_argument("--rsi-long-max", type=float, default=40.0)
    parser.add_argument("--rsi-short-min", type=float, default=60.0)
    parser.add_argument("--adx-max", type=float, default=20.0)
    parser.add_argument("--ema-slope-period", type=int, default=50)
    parser.add_argument("--ema-slope-lookback", type=int, default=1)
    parser.add_argument("--slope-atr-max", type=float, default=0.20)
    parser.add_argument("--atr-ratio-min", type=float, default=0.65)
    parser.add_argument("--atr-ratio-max", type=float, default=1.65)
    parser.add_argument("--atr-median-bars", type=int, default=5760)
    parser.add_argument("--max-spread-points", type=float, default=120.0)
    parser.add_argument("--max-hold-bars", type=int, default=36)
    parser.add_argument(
        "--trend-conflict-momentum-threshold",
        type=float,
        default=0.0,
        help="Block BBRSI counter-trend entries when same-direction momentum conflicts; 0 disables.",
    )
    parser.add_argument(
        "--trend-conflict-ema-buffer-atr",
        type=float,
        default=0.0,
        help="ATR buffer beyond EMA required for BBRSI trend-conflict block.",
    )
    parser.add_argument("--once", action="store_true", help="Run exactly one bar evaluation for smoke tests.")
    parser.add_argument("--self-test", action="store_true", help="Run local indicator sanity checks and exit.")
    return parser


def _run_self_test() -> None:
    values = [float(i) for i in range(1, 80)]
    mid, upper, lower = BbRsiRangingMT5Strategy._bollinger(values, 20, 2.0)
    rsi = BbRsiRangingMT5Strategy._rsi(values, 14)
    ema = BbRsiRangingMT5Strategy._ema_series(values, 10)
    assert lower < mid < upper
    assert rsi > 99.0
    assert not math.isnan(ema[-1])
    print("self-test ok")


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        _run_self_test()
        return
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    config = BbRsiConfig(
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
        terminal_path=args.terminal_path.strip() or None,
        log_file=args.log_file.strip() or None,
        state_path=args.state_path.strip() or None,
        bb_period=int(args.bb_period),
        bb_std=float(args.bb_std),
        rsi_period=int(args.rsi_period),
        rsi_long_max=float(args.rsi_long_max),
        rsi_short_min=float(args.rsi_short_min),
        adx_max=float(args.adx_max),
        ema_slope_period=int(args.ema_slope_period),
        ema_slope_lookback=int(args.ema_slope_lookback),
        slope_atr_max=float(args.slope_atr_max),
        atr_ratio_min=float(args.atr_ratio_min),
        atr_ratio_max=float(args.atr_ratio_max),
        atr_median_bars=int(args.atr_median_bars),
        max_spread_points=float(args.max_spread_points),
        max_hold_bars=int(args.max_hold_bars),
        trend_conflict_momentum_threshold=float(args.trend_conflict_momentum_threshold),
        trend_conflict_ema_buffer_atr=float(args.trend_conflict_ema_buffer_atr),
        once=bool(args.once),
        self_test=bool(args.self_test),
    )
    strategy = BbRsiRangingMT5Strategy(config)
    if args.once:
        strategy.run_once()
    else:
        strategy.run()


if __name__ == "__main__":
    main()
