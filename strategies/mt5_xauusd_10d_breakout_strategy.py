#!/usr/bin/env python3.14
"""10-day XAUUSD high-return breakout campaign strategy.

Design:
- Trade XAUUSD only through the existing pymt5linux bridge.
- Optimize for asymmetric payoff over a short 10-day campaign.
- Do not optimize win rate; require strong volatility expansion and let winners run.
- Enforce hard peak-to-equity and campaign-start drawdown limits near 5%.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from mt5_doomsday_strategy import (
    DoomsdayMT5Strategy,
    MarketSnapshot,
    PositionState,
    StrategyConfig as BaseStrategyConfig,
    build_parser as build_base_parser,
)


LOGGER = logging.getLogger("mt5_xauusd_10d_breakout_strategy")
STATE_PATH_DEFAULT = "/home/chain4655/Documents/Projects/MT5/auto_quant/state/xauusd_10d_breakout_state.json"
LOG_PATH_DEFAULT = "/home/chain4655/Documents/Projects/MT5/auto_quant/logs/xauusd_10d_breakout_strategy.log"


@dataclasses.dataclass(frozen=True)
class StrategyConfig(BaseStrategyConfig):
    campaign_days: int
    max_campaign_drawdown_pct: float
    entry_lookback: int
    ema_fast: int
    ema_slow: int
    momentum_lookback: int
    min_body_atr: float
    min_close_location: float
    max_spread_points: float
    breakeven_r: float
    trail_start_r: float
    trail_atr: float


@dataclasses.dataclass
class CampaignState:
    campaign_start_time: Optional[str] = None
    campaign_start_equity: Optional[float] = None
    day_key: Optional[str] = None
    day_start_equity: Optional[float] = None
    max_equity_seen: Optional[float] = None
    stopped_reason: str = ""


class XAUUSD10DayBreakoutStrategy(DoomsdayMT5Strategy):
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.config: StrategyConfig = config
        self._campaign_state = self._load_campaign_state()

    def _load_campaign_state(self) -> CampaignState:
        if not self._state_path:
            return CampaignState()
        path = Path(self._state_path)
        if not path.exists():
            return CampaignState()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return CampaignState(
                campaign_start_time=data.get("campaign_start_time"),
                campaign_start_equity=data.get("campaign_start_equity"),
                day_key=data.get("day_key"),
                day_start_equity=data.get("day_start_equity"),
                max_equity_seen=data.get("max_equity_seen"),
                stopped_reason=str(data.get("stopped_reason", "")),
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            self._log_file("Campaign state load failed, starting fresh: %s", exc)
            return CampaignState()

    def _save_campaign_state(self) -> None:
        if not self._state_path:
            return
        payload = self._campaign_payload()
        try:
            path = Path(self._state_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except OSError as exc:
            LOGGER.warning("Campaign state save failed: %s", exc)

    def _write_state(self, payload: Dict[str, Any]) -> None:
        if not self._state_path:
            return
        state_payload = self._campaign_payload()
        state_payload["runtime"] = payload
        try:
            path = Path(self._state_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(state_payload, indent=2, sort_keys=True), encoding="utf-8")
        except OSError as exc:
            LOGGER.warning("State write failed: %s", exc)

    def _campaign_payload(self) -> Dict[str, Any]:
        return dataclasses.asdict(self._campaign_state)

    def _seed_equity(self) -> None:
        equity = self._get_equity()
        now = dt.datetime.now().isoformat(sep=" ", timespec="seconds")
        today = dt.date.today().isoformat()

        if self._campaign_state.campaign_start_time is None:
            self._campaign_state.campaign_start_time = now
        if self._campaign_state.campaign_start_equity is None:
            self._campaign_state.campaign_start_equity = equity
        if self._campaign_state.day_key != today:
            self._campaign_state.day_key = today
            self._campaign_state.day_start_equity = equity
        if self._campaign_state.max_equity_seen is None:
            self._campaign_state.max_equity_seen = equity

        self._initial_equity = float(self._campaign_state.campaign_start_equity)
        self._session_start_equity = float(self._campaign_state.day_start_equity or equity)
        self._max_equity_seen = float(self._campaign_state.max_equity_seen or equity)
        self._save_campaign_state()
        self._log_file("10D campaign seeded: start_equity=%.2f current_equity=%.2f", self._initial_equity, equity)
        LOGGER.info("10D campaign seeded: start_equity=%.2f current_equity=%.2f", self._initial_equity, equity)

    def _risk_guard(self) -> None:
        equity = self._get_equity()
        now = dt.datetime.now()
        today = now.date().isoformat()

        if self._campaign_state.day_key != today:
            self._campaign_state.day_key = today
            self._campaign_state.day_start_equity = equity

        start_equity = float(self._campaign_state.campaign_start_equity or equity)
        day_start_equity = float(self._campaign_state.day_start_equity or equity)
        peak_equity = max(float(self._campaign_state.max_equity_seen or equity), equity)
        self._campaign_state.max_equity_seen = peak_equity
        self._initial_equity = start_equity
        self._session_start_equity = day_start_equity
        self._max_equity_seen = peak_equity

        dd_from_peak = max(0.0, 1.0 - equity / max(peak_equity, 1e-9))
        dd_from_start = max(0.0, 1.0 - equity / max(start_equity, 1e-9))
        daily_dd = max(0.0, 1.0 - equity / max(day_start_equity, 1e-9))
        elapsed_days = self._campaign_elapsed_days(now)

        self._log_file(
            "Risk | equity=%.2f start=%.2f peak=%.2f campaign_dd=%.2f%% peak_dd=%.2f%% daily_dd=%.2f%% day=%.2f/%d",
            equity,
            start_equity,
            peak_equity,
            dd_from_start * 100.0,
            dd_from_peak * 100.0,
            daily_dd * 100.0,
            elapsed_days,
            self.config.campaign_days,
        )
        self._save_campaign_state()

        limit = min(self.config.max_campaign_drawdown_pct, self.config.max_drawdown_pct)
        if dd_from_peak >= limit or dd_from_start >= limit:
            reason = (
                f"Campaign drawdown stop: start_dd={dd_from_start * 100.0:.2f}% "
                f"peak_dd={dd_from_peak * 100.0:.2f}% limit={limit * 100.0:.2f}%"
            )
            self._stop_campaign(reason, exit_code=1)

        if daily_dd >= self.config.max_daily_loss_pct:
            reason = f"Daily drawdown stop: {daily_dd * 100.0:.2f}% >= {self.config.max_daily_loss_pct * 100.0:.2f}%"
            self._stop_campaign(reason, exit_code=1)

        if elapsed_days >= float(self.config.campaign_days):
            self._stop_campaign(f"Campaign completed: {elapsed_days:.2f} days", exit_code=0)

    def _campaign_elapsed_days(self, now: dt.datetime) -> float:
        raw = self._campaign_state.campaign_start_time
        if not raw:
            return 0.0
        try:
            start = dt.datetime.fromisoformat(raw)
        except ValueError:
            return 0.0
        return max(0.0, (now - start).total_seconds() / 86400.0)

    def _stop_campaign(self, reason: str, exit_code: int) -> None:
        self._campaign_state.stopped_reason = reason
        self._save_campaign_state()
        self._log_file(reason)
        LOGGER.warning(reason)
        self.close_all_positions()
        raise SystemExit(exit_code)

    def _build_snapshot(self) -> MarketSnapshot:
        need = max(
            self.config.entry_lookback,
            self.config.ema_slow,
            self.config.slow_sma,
            self.config.atr_period,
            self.config.momentum_lookback,
        ) + 10
        bars = self._fetch_bars(min_count=need)
        if len(bars) < need:
            raise RuntimeError(
                f"Not enough bars for signal calculation: got={len(bars)} need={need} lookback={self.config.lookback_bars}"
            )

        last_closed = bars[-2]
        closed_bars = bars[:-1]
        closes = [bar["close"] for bar in closed_bars]
        highs = [bar["high"] for bar in closed_bars]
        lows = [bar["low"] for bar in closed_bars]
        opens = [bar["open"] for bar in closed_bars]

        atr = self._atr(closed_bars, self.config.atr_period)
        fast_ema = self._ema(closes, self.config.ema_fast)
        slow_ema = self._ema(closes, self.config.ema_slow)
        momentum = self._momentum_atr(closes, atr)
        atr_pct = atr / max(float(last_closed["close"]), 1e-9)
        regime_lookback = min(30, len(highs), len(lows))
        range_atr = (max(highs[-regime_lookback:]) - min(lows[-regime_lookback:])) / max(atr, 1e-9)
        candle_range = max(float(last_closed["high"]) - float(last_closed["low"]), 0.0)
        spike_atr = candle_range / max(atr, 1e-9)
        close_location = (
            (float(last_closed["close"]) - float(last_closed["low"])) / candle_range
            if candle_range > 0.0
            else 0.5
        )
        breakout_atr = self._breakout_atr(closes, highs, lows, atr)
        high_volatility = self._is_high_volatility_regime(atr_pct, range_atr, spike_atr)
        score = self._score_campaign_signal(
            closes=closes,
            opens=opens,
            highs=highs,
            lows=lows,
            atr=atr,
            fast_ema=fast_ema,
            slow_ema=slow_ema,
            momentum=momentum,
            close_location=close_location,
        )
        signal = self._decide_signal(score)
        if self.config.high_vol_only and not high_volatility:
            signal = "NONE"

        return MarketSnapshot(
            bar_time=self._bar_time(last_closed),
            close=float(last_closed["close"]),
            high=float(last_closed["high"]),
            low=float(last_closed["low"]),
            atr=atr,
            fast_sma=fast_ema,
            slow_sma=slow_ema,
            momentum=momentum,
            score=score,
            signal=signal,
            atr_pct=atr_pct,
            range_atr=range_atr,
            spike_atr=spike_atr,
            breakout_atr=breakout_atr,
            close_location=close_location,
            high_volatility=high_volatility,
        )

    def _score_campaign_signal(
        self,
        closes: Sequence[float],
        opens: Sequence[float],
        highs: Sequence[float],
        lows: Sequence[float],
        atr: float,
        fast_ema: float,
        slow_ema: float,
        momentum: float,
        close_location: float,
    ) -> float:
        last_close = closes[-1]
        last_open = opens[-1]
        lookback = max(8, min(self.config.entry_lookback, len(highs) - 1))
        recent_high = max(highs[-lookback - 1 : -1])
        recent_low = min(lows[-lookback - 1 : -1])
        body_atr = abs(last_close - last_open) / max(atr, 1e-9)

        if body_atr < self.config.min_body_atr:
            return 0.0

        trend = 0.0
        if last_close > fast_ema > slow_ema:
            trend = 0.45
        elif last_close < fast_ema < slow_ema:
            trend = -0.45
        elif last_close > slow_ema:
            trend = 0.12
        elif last_close < slow_ema:
            trend = -0.12

        up_break = self._clamp((last_close - recent_high) / max(atr, 1e-9), -2.0, 2.0)
        down_break = self._clamp((recent_low - last_close) / max(atr, 1e-9), -2.0, 2.0)
        breakout = up_break * 0.55 - down_break * 0.55
        momentum_component = self._clamp(momentum, -2.5, 2.5) * 0.35

        location_component = 0.0
        if close_location >= self.config.min_close_location:
            location_component = 0.18
        elif close_location <= (1.0 - self.config.min_close_location):
            location_component = -0.18

        bias_component = (self.config.long_bias - 0.5) * 0.10
        return self._clamp(trend + breakout + momentum_component + location_component + bias_component, -1.8, 1.8)

    def _high_vol_entry_ok(self, snapshot: MarketSnapshot) -> bool:
        if not super()._high_vol_entry_ok(snapshot):
            return False
        other_symbol_positions = [
            pos
            for pos in self._symbol_positions()
            if int(getattr(pos, "magic", 0)) != int(self.config.magic)
        ]
        if other_symbol_positions:
            self._log_file("ENTRY_FILTERED other XAUUSD positions present: %d", len(other_symbol_positions))
            return False
        spread = self._spread_points()
        if spread > self.config.max_spread_points:
            self._log_file("ENTRY_FILTERED spread %.1f > %.1f", spread, self.config.max_spread_points)
            return False
        return True

    def _maybe_trail(self, snapshot: MarketSnapshot, pos: PositionState) -> None:
        if snapshot.atr <= 0 or pos.sl <= 0:
            return
        ask, bid = self._tick_prices()
        current_price = bid if pos.type == self.mt5.POSITION_TYPE_BUY else ask
        profit_move = (
            current_price - pos.price_open
            if pos.type == self.mt5.POSITION_TYPE_BUY
            else pos.price_open - current_price
        )
        initial_r = abs(pos.price_open - pos.sl)
        if initial_r <= self._point():
            initial_r = snapshot.atr * self.config.stop_atr

        if profit_move >= initial_r * self.config.breakeven_r:
            if pos.type == self.mt5.POSITION_TYPE_BUY:
                be_sl = pos.price_open + self._point() * 20.0
                if be_sl > pos.sl:
                    self._modify_position(pos, sl=self._round_to_digits(be_sl, self._digits()), tp=pos.tp)
            else:
                be_sl = pos.price_open - self._point() * 20.0
                if be_sl < pos.sl:
                    self._modify_position(pos, sl=self._round_to_digits(be_sl, self._digits()), tp=pos.tp)

        if profit_move < initial_r * self.config.trail_start_r:
            return

        trail_distance = snapshot.atr * self.config.trail_atr
        if pos.type == self.mt5.POSITION_TYPE_BUY:
            new_sl = current_price - trail_distance
            if new_sl > pos.sl:
                self._modify_position(pos, sl=self._round_to_digits(new_sl, self._digits()), tp=pos.tp)
        else:
            new_sl = current_price + trail_distance
            if new_sl < pos.sl:
                self._modify_position(pos, sl=self._round_to_digits(new_sl, self._digits()), tp=pos.tp)

    def _spread_points(self) -> float:
        ask, bid = self._tick_prices()
        return abs(ask - bid) / max(self._point(), 1e-9)

    def _ema(self, values: Sequence[float], period: int) -> float:
        if not values:
            return 0.0
        if len(values) < period:
            return statistics.fmean(values)
        alpha = 2.0 / (float(period) + 1.0)
        ema = statistics.fmean(values[:period])
        for value in values[period:]:
            ema = alpha * value + (1.0 - alpha) * ema
        return ema

    def _momentum_atr(self, closes: Sequence[float], atr: float) -> float:
        lookback = max(1, min(self.config.momentum_lookback, len(closes) - 1))
        return self._clamp((closes[-1] - closes[-1 - lookback]) / max(atr, 1e-9), -3.0, 3.0)


def build_parser() -> argparse.ArgumentParser:
    parser = build_base_parser()
    parser.description = "10-day XAUUSD high-return breakout campaign strategy."
    parser.set_defaults(
        risk_pct=0.009,
        stop_atr=2.6,
        reward_multiple=5.0,
        tp_min_usd=30.0,
        tp_max_usd=120.0,
        long_bias=0.56,
        trend_threshold=0.78,
        roll_trigger_pct=0.12,
        cooldown_minutes=15,
        max_leverage=8.0,
        max_drawdown_pct=0.048,
        max_daily_loss_pct=0.018,
        max_lots=3.0,
        magic=210510,
        deviation=50,
        loop_seconds=8,
        lookback_bars=180,
        atr_period=14,
        fast_sma=8,
        slow_sma=34,
        high_vol_only=True,
        high_vol_atr_pct=0.0017,
        high_vol_range_atr=3.8,
        high_vol_breakout_lookback=18,
        high_vol_min_momentum=0.75,
        high_vol_spike_atr=2.4,
        high_vol_min_breakout_atr=0.32,
        high_vol_min_close_location=0.66,
        log_file=LOG_PATH_DEFAULT,
        state_path=STATE_PATH_DEFAULT,
    )
    parser.add_argument("--campaign-days", type=int, default=10)
    parser.add_argument("--max-campaign-drawdown-pct", type=float, default=0.048)
    parser.add_argument("--entry-lookback", type=int, default=18)
    parser.add_argument("--ema-fast", type=int, default=9)
    parser.add_argument("--ema-slow", type=int, default=34)
    parser.add_argument("--momentum-lookback", type=int, default=4)
    parser.add_argument("--min-body-atr", type=float, default=0.20)
    parser.add_argument("--min-close-location", type=float, default=0.66)
    parser.add_argument("--max-spread-points", type=float, default=45.0)
    parser.add_argument("--breakeven-r", type=float, default=1.05)
    parser.add_argument("--trail-start-r", type=float, default=2.2)
    parser.add_argument("--trail-atr", type=float, default=1.15)
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
        terminal_path=terminal_path,
        log_file=args.log_file.strip() or None,
        state_path=args.state_path.strip() or None,
        campaign_days=int(args.campaign_days),
        max_campaign_drawdown_pct=float(args.max_campaign_drawdown_pct),
        entry_lookback=int(args.entry_lookback),
        ema_fast=int(args.ema_fast),
        ema_slow=int(args.ema_slow),
        momentum_lookback=int(args.momentum_lookback),
        min_body_atr=float(args.min_body_atr),
        min_close_location=float(args.min_close_location),
        max_spread_points=float(args.max_spread_points),
        breakeven_r=float(args.breakeven_r),
        trail_start_r=float(args.trail_start_r),
        trail_atr=float(args.trail_atr),
    )
    strategy = XAUUSD10DayBreakoutStrategy(config)
    strategy.run()


if __name__ == "__main__":
    main()
