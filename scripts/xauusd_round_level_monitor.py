#!/usr/bin/env python3.14
"""Monitor XAUUSD round-number crossings and test reversal strength.

The script is read-only. It connects to the live MT5 bridge through pymt5linux,
fetches recent bars, and checks whether crossings of round levels (default: every
100.0 price points) are followed by strong opposite moves.

It is intended to help falsify the claim that crossings of integer levels such as
4500 / 4800 reliably trigger a strong reversal burst.
"""
from __future__ import annotations

import math
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path("/home/chain4655/Documents/Projects/MT5")
PYMT5_SITE = Path("/home/chain4655/.openharness-venv/lib64/python3.14/site-packages")
if str(PYMT5_SITE) not in sys.path:
    sys.path.insert(0, str(PYMT5_SITE))

try:
    from pymt5linux import MetaTrader5  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"pymt5linux import failed: {exc}")


@dataclass(frozen=True)
class CrossingEvent:
    level: float
    bar_time: datetime
    direction: str
    price_before: float
    price_after: float
    reversal_6: float
    followthrough_6: float
    reversal_12: float
    followthrough_12: float


@dataclass(frozen=True)
class RoundLevelSummary:
    level: float
    crossings: int
    bullish_crosses: int
    bearish_crosses: int
    median_reversal_6: float
    median_followthrough_6: float
    median_reversal_12: float
    median_followthrough_12: float
    reversal_dominance_rate_6: float
    reversal_dominance_rate_12: float


def _resolve_timeframe(mt5: Any, timeframe: str) -> int:
    key = timeframe.strip().upper()
    mapping = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return mapping[key]


def _fresh_client(host: str = "127.0.0.1", port: int = 18812) -> MetaTrader5:
    client = MetaTrader5(host=host, port=port)
    ok = client.initialize()
    if not ok:
        raise RuntimeError(f"initialize() failed: {client.last_error()}")
    return client


def _normalize_rate(row: Any) -> Dict[str, float]:
    if hasattr(row, "_asdict"):
        data = dict(row._asdict())
        return {
            "time": float(data["time"]),
            "open": float(data["open"]),
            "high": float(data["high"]),
            "low": float(data["low"]),
            "close": float(data["close"]),
        }
    if isinstance(row, dict):
        return {
            "time": float(row["time"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }
    return {
        "time": float(row["time"]),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
    }


def _bar_time_from_epoch(epoch_seconds: float) -> datetime:
    return datetime.fromtimestamp(int(epoch_seconds))


def _load_bars(symbol: str, timeframe: str, count: int, host: str = "127.0.0.1", port: int = 18812) -> List[Dict[str, float]]:
    client = _fresh_client(host=host, port=port)
    try:
        tf = _resolve_timeframe(client, timeframe)
        bars = client.copy_rates_from_pos(symbol, tf, 0, count)
        if bars is None:
            raise RuntimeError(f"copy_rates_from_pos failed: {client.last_error()}")
        return [_normalize_rate(row) for row in list(bars)]
    finally:
        client.shutdown()


def _load_tick(symbol: str, host: str = "127.0.0.1", port: int = 18812) -> Tuple[float, float]:
    client = _fresh_client(host=host, port=port)
    try:
        tick = client.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError("symbol_info_tick returned None")
        return float(getattr(tick, "bid", 0.0) or 0.0), float(getattr(tick, "ask", 0.0) or 0.0)
    finally:
        client.shutdown()


def _crossings_for_level(bars: Sequence[Dict[str, float]], level: float) -> List[Tuple[int, str]]:
    hits: List[Tuple[int, str]] = []
    for i in range(1, len(bars)):
        prev_close = bars[i - 1]["close"]
        close = bars[i]["close"]
        if prev_close < level <= close:
            hits.append((i, "UP"))
        elif prev_close > level >= close:
            hits.append((i, "DOWN"))
    return hits


def _reaction_after_cross(
    bars: Sequence[Dict[str, float]],
    idx: int,
    level: float,
    direction: str,
    lookahead: int,
) -> Tuple[float, float]:
    future = bars[idx + 1 : idx + 1 + lookahead]
    if not future:
        return 0.0, 0.0
    highs = [bar["high"] for bar in future]
    lows = [bar["low"] for bar in future]
    if direction == "UP":
        reversal = max(0.0, level - min(lows))
        follow = max(0.0, max(highs) - level)
    else:
        reversal = max(0.0, max(highs) - level)
        follow = max(0.0, level - min(lows))
    return reversal, follow


def _summarize_level(bars: Sequence[Dict[str, float]], level: float) -> Optional[RoundLevelSummary]:
    events = _crossings_for_level(bars, level)
    if not events:
        return None

    evs: List[CrossingEvent] = []
    for idx, direction in events:
        reversal_6, follow_6 = _reaction_after_cross(bars, idx, level, direction, 6)
        reversal_12, follow_12 = _reaction_after_cross(bars, idx, level, direction, 12)
        evs.append(
            CrossingEvent(
                level=level,
                bar_time=_bar_time_from_epoch(bars[idx]["time"]),
                direction=direction,
                price_before=bars[idx - 1]["close"],
                price_after=bars[idx]["close"],
                reversal_6=reversal_6,
                followthrough_6=follow_6,
                reversal_12=reversal_12,
                followthrough_12=follow_12,
            )
        )

    reversal_dom_6 = sum(1 for e in evs if e.reversal_6 > e.followthrough_6) / len(evs)
    reversal_dom_12 = sum(1 for e in evs if e.reversal_12 > e.followthrough_12) / len(evs)
    rev6 = [e.reversal_6 for e in evs]
    fol6 = [e.followthrough_6 for e in evs]
    rev12 = [e.reversal_12 for e in evs]
    fol12 = [e.followthrough_12 for e in evs]
    return RoundLevelSummary(
        level=level,
        crossings=len(evs),
        bullish_crosses=sum(1 for e in evs if e.direction == "UP"),
        bearish_crosses=sum(1 for e in evs if e.direction == "DOWN"),
        median_reversal_6=statistics.median(rev6),
        median_followthrough_6=statistics.median(fol6),
        median_reversal_12=statistics.median(rev12),
        median_followthrough_12=statistics.median(fol12),
        reversal_dominance_rate_6=reversal_dom_6,
        reversal_dominance_rate_12=reversal_dom_12,
    )


def _nearest_round_levels(price: float, step: float, radius_steps: int = 2) -> List[float]:
    base = math.floor(price / step) * step
    levels = [base + i * step for i in range(-radius_steps, radius_steps + 1)]
    return [round(x, 2) for x in levels if x > 0]


def main() -> int:
    symbol = "XAUUSD"
    timeframe = "M5"
    count = 600
    step = 100.0
    host = "127.0.0.1"
    port = 18812

    bars = _load_bars(symbol, timeframe, count=count, host=host, port=port)
    bid, ask = _load_tick(symbol, host=host, port=port)
    current = (bid + ask) / 2.0 if bid > 0 and ask > 0 else bars[-1]["close"]
    levels = _nearest_round_levels(current, step, radius_steps=3)
    all_levels = [lvl for lvl in levels if min(bar["low"] for bar in bars) <= lvl <= max(bar["high"] for bar in bars)]

    summaries = []
    for lvl in all_levels:
        summary = _summarize_level(bars, lvl)
        if summary is not None:
            summaries.append(summary)

    print(f"symbol={symbol} timeframe={timeframe} bars={len(bars)} current={current:.2f} bid={bid:.2f} ask={ask:.2f}")
    print(f"analyzed_round_levels={len(all_levels)} step={step:.0f}")
    print("---")
    if not summaries:
        print("No round-level crossings found in the sampled history.")
        return 0

    for s in summaries:
        verdict_6 = "reversal" if s.reversal_dominance_rate_6 >= 0.5 else "follow-through"
        verdict_12 = "reversal" if s.reversal_dominance_rate_12 >= 0.5 else "follow-through"
        print(
            f"level={s.level:.0f} crossings={s.crossings} up={s.bullish_crosses} down={s.bearish_crosses} "
            f"med_rev6={s.median_reversal_6:.2f} med_follow6={s.median_followthrough_6:.2f} "
            f"dom6={s.reversal_dominance_rate_6:.2%} verdict6={verdict_6} "
            f"med_rev12={s.median_reversal_12:.2f} med_follow12={s.median_followthrough_12:.2f} "
            f"dom12={s.reversal_dominance_rate_12:.2%} verdict12={verdict_12}"
        )

    all_dom_6 = statistics.mean(s.reversal_dominance_rate_6 for s in summaries)
    all_dom_12 = statistics.mean(s.reversal_dominance_rate_12 for s in summaries)
    print("---")
    print(f"overall_reversal_dominance_6={all_dom_6:.2%}")
    print(f"overall_reversal_dominance_12={all_dom_12:.2%}")
    if all_dom_6 >= 0.60 and all_dom_12 >= 0.60:
        print("preliminary_read: current data weakly supports strong mean-reversion after round-number crossings")
    else:
        print("preliminary_read: current data does NOT strongly support a reliable strong reversal effect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
