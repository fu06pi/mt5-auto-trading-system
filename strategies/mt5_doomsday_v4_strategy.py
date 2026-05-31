"""MT5 Doomsday V4: ATR-proportional TP at reward_multiple × ATR.
Fix over V2: TP was fixed USD($24-72), now ATR-based for proper 1:2 R:R.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

from mt5_doomsday_strategy import DoomsdayMT5Strategy, StrategyConfig, build_parser

LOGGER = logging.getLogger("mt5_doomsday_v4_strategy")


class DoomsdayV4Strategy(DoomsdayMT5Strategy):
    def _build_sl_tp(self, direction: str, price: float, atr: float, score: float) -> Tuple[float, float]:
        sl_distance = atr * self.config.stop_atr
        tp_distance = sl_distance * self.config.reward_multiple
        point = self._point()
        digits = self._digits()

        if direction == "BUY":
            sl = price - sl_distance
            tp = price + tp_distance
        else:
            sl = price + sl_distance
            tp = price - tp_distance

        sl = self._round_to_digits(sl, digits)
        tp = self._round_to_digits(tp, digits)
        if abs(price - sl) < point:
            sl = price - point if direction == "BUY" else price + point
        if abs(price - tp) < point:
            tp = price + point if direction == "BUY" else price - point
        return sl, tp


def make_v4_parser() -> argparse.ArgumentParser:
    p = build_parser()
    p.set_defaults(
        reward_multiple=2.0,
        tp_min_usd=0.0,
        tp_max_usd=0.0,
    )
    return p


def main() -> None:
    args = make_v4_parser().parse_args()
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
    )

    strategy = DoomsdayV4Strategy(config)
    strategy.run()


if __name__ == "__main__":
    main()
