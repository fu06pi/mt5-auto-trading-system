#!/usr/bin/env python3.14
"""Inverse-tight XAUUSD main-strategy variant for MT5.

This is a branch/variant of ``mt5_xauusd_trend_strategy.py`` with three deliberate
changes for isolated research/dry-run use:

1. Main trend entries are inverted: a tightened BUY setup enters SELL, and a
   tightened SELL setup enters BUY.
2. Entry threshold is tightened versus the base trend strategy.
3. Hard account risk guards are disabled in this variant: daily DD, total DD,
   profit-target, and best-day concentration no longer pause/close/exit.

Live ``active_plan.json`` is not touched by this file.  If this is ever attached
to a live sleeve, use a distinct magic number, state file, log file, and order
comment so it cannot collide with the production strategy.
"""

from __future__ import annotations

import logging
from typing import Sequence, Tuple

import mt5_xauusd_trend_strategy as base


LOGGER = logging.getLogger("xauusd_inverse_tight_main_strategy")


class XAUUSDInverseTightMainStrategy(base.XAUUSDTrendStrategy):
    """Tight-entry inverse of the base XAUUSD main trend strategy."""

    # Tighten entry: require the base signal to exceed the configured threshold
    # by both a multiplier and an absolute buffer.  With the current live
    # threshold 0.35, this becomes max(0.525, 0.50) = 0.525.
    INVERSE_ENTRY_THRESHOLD_MULT = 1.50
    INVERSE_ENTRY_THRESHOLD_BUFFER = 0.15

    def _risk_guard(self, snapshot: base.MarketSnapshot) -> None:
        """Observe account equity but do not enforce hard pause/close guards.

        Disabled here by design for the inverse branch only:
        - daily drawdown stop
        - total drawdown stop
        - profit target stop
        - best-day concentration pause

        Kept: state bookkeeping/logging so diagnostics still show the would-be
        DD/profit progress.  Position sizing, SL/TP construction, spread filter,
        session filter, position caps, and broker margin checks remain governed
        by the normal entry/order code and CLI values.
        """
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

        if self.state.paused:
            self.state.paused = False
            self.state.paused_reason = ""

        self._log(
            "Risk guard DISABLED | equity=%.2f day_dd=%.2f%% total_dd=%.2f%% profit=%.2f%% best_share=%.2f%%",
            equity,
            daily_dd * 100.0,
            total_dd * 100.0,
            profit_progress * 100.0,
            self._best_day_share() * 100.0,
        )
        self._save_state()

    def _decide_signal(self, score: float, htf_signal: str, spread_points: float) -> str:
        """Invert the tightened primary trend signal.

        Base main signal:
        - BULL + strong positive score => BUY
        - BEAR + strong negative score => SELL

        This variant:
        - BULL + stronger positive score => SELL
        - BEAR + stronger negative score => BUY
        """
        if spread_points > self.config.max_spread_points:
            return "NONE"

        threshold = max(
            float(self.config.trend_threshold) * self.INVERSE_ENTRY_THRESHOLD_MULT,
            float(self.config.trend_threshold) + self.INVERSE_ENTRY_THRESHOLD_BUFFER,
        )
        if htf_signal == "BULL" and score >= threshold:
            return "SELL"
        if htf_signal == "BEAR" and score <= -threshold:
            return "BUY"
        return "NONE"

    def _false_breakout_reversal_signal(
        self,
        closed_bars: Sequence[dict],
        atr: float,
        htf_signal: str,
        spread_points: float,
    ) -> Tuple[str, str]:
        """Disable complement/overlay entries; this branch is main-strategy only."""
        return "NONE", "disabled_inverse_main_only"

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
        """Do not apply base lag-reversal blocker after we intentionally invert."""
        return False

    def _entry_order_comment(self) -> str:
        configured = str(self.config.order_comment or "").strip()
        if configured:
            return configured[:31]
        return "xauusd-inverse-tight" if self.config.live else "inverse-tight-dryrun"


# Reuse the mature parser/config builder from the base strategy, but swap the
# class at instantiation time.  This keeps CLI compatibility with active-plan
# commands without duplicating 100+ config assignments.
def main() -> None:
    base.XAUUSDTrendStrategy = XAUUSDInverseTightMainStrategy
    base.main()


if __name__ == "__main__":
    main()
