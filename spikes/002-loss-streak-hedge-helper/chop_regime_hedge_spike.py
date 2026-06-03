#!/usr/bin/env python3
"""Spike: chop-regime hedge helper for XAUUSD trend strategy.

Research-only. Reads cached broker M5 bars and simulates:
- baseline current trend sleeve;
- trend + direct chop-regime helper sleeve;
- trend pause/scale-down inside chop as a negative/defensive comparison.

No MT5 connection, no orders, no active_plan edits.
"""
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import math
import statistics
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path("/home/chain4655/Documents/Projects/MT5")
sys.path.insert(0, str(ROOT))

from research_xauusd_trend_plus_complement_backtest import (  # noqa: E402
    CONTRACT_SIZE,
    INITIAL_EQUITY,
    POINT,
    Params,
    Trade,
    build_snapshot,
    costs,
    mean,
    stdev,
)
from research_xauusd_trend_plus_complement_long_cache_backtest import (  # noqa: E402
    CSV_PATH,
    aggregate_bars,
    load_m5_csv,
)

OUT_DIR = ROOT / "spikes/002-loss-streak-hedge-helper/results"


@dataclasses.dataclass(frozen=True)
class ChopParams:
    name: str
    action: str  # hedge_opposite, fade_breakout, pause_trend, scale_trend
    adx_max: float
    efficiency_max: float
    atr_ratio_max: float
    slope_atr_max: float
    alternation_min: float
    min_score: float
    risk_mult: float
    reward_multiple: float
    max_hold_bars: int
    require_primary_signal: bool = True
    min_chop_points: int = 3


@dataclasses.dataclass(frozen=True)
class ChopState:
    is_chop: bool
    score: float
    adx: float
    efficiency: float
    atr_ratio: float
    slope_atr: float
    alternation: float
    reason: str


@dataclasses.dataclass
class OpenPosition:
    sleeve: str
    side: str
    entry: float
    sl: float
    orig_sl: float
    tp: float
    volume: float
    entry_time: dt.datetime
    entry_i: int
    atr: float
    score: float
    htf_signal: str
    session: str
    context: str


def round_volume(lots: float) -> float:
    return max(0.0, math.floor(lots / 0.01) * 0.01)


def sized_volume(equity: float, risk_pct: float, max_lots: float, entry: float, sl: float) -> float:
    risk_per_lot = abs(entry - sl) * CONTRACT_SIZE
    if risk_per_lot <= 0:
        return 0.0
    return round_volume(min(max_lots, (equity * risk_pct) / risk_per_lot))


def true_ranges(bars: Sequence[Any]) -> List[float]:
    out: List[float] = []
    for i in range(1, len(bars)):
        curr = bars[i]
        prev = bars[i - 1]
        out.append(max(curr.high - curr.low, abs(curr.high - prev.close), abs(curr.low - prev.close)))
    return out


def calculate_adx(bars: Sequence[Any], period: int = 14) -> float:
    if len(bars) < period * 2 + 2:
        return 50.0
    plus_dm: List[float] = []
    minus_dm: List[float] = []
    trs: List[float] = []
    for i in range(1, len(bars)):
        up_move = bars[i].high - bars[i - 1].high
        down_move = bars[i - 1].low - bars[i].low
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
        trs.append(max(bars[i].high - bars[i].low, abs(bars[i].high - bars[i - 1].close), abs(bars[i].low - bars[i - 1].close)))
    dxs: List[float] = []
    for j in range(period, len(trs) + 1):
        tr_sum = sum(trs[j - period:j])
        if tr_sum <= 0:
            continue
        plus_di = 100.0 * sum(plus_dm[j - period:j]) / tr_sum
        minus_di = 100.0 * sum(minus_dm[j - period:j]) / tr_sum
        denom = plus_di + minus_di
        if denom <= 0:
            continue
        dxs.append(100.0 * abs(plus_di - minus_di) / denom)
    return mean(dxs[-period:]) if dxs else 50.0


def range_efficiency(bars: Sequence[Any], window: int = 48) -> float:
    if len(bars) < window + 1:
        return 1.0
    recent = bars[-window - 1:]
    net_move = abs(recent[-1].close - recent[0].close)
    path = sum(abs(recent[i].close - recent[i - 1].close) for i in range(1, len(recent)))
    return net_move / max(path, POINT)


def atr_ratio(bars: Sequence[Any], short_period: int = 14, long_period: int = 96) -> float:
    trs = true_ranges(bars)
    if len(trs) < long_period:
        return 1.0
    short_atr = mean(trs[-short_period:])
    long_atr = mean(trs[-long_period:])
    return short_atr / max(long_atr, POINT)


def alternating_signal_rate(signals: Sequence[str]) -> float:
    directional = [sig for sig in signals if sig in {"BUY", "SELL"}]
    if len(directional) < 4:
        return 0.0
    flips = sum(1 for i in range(1, len(directional)) if directional[i] != directional[i - 1])
    return flips / max(len(directional) - 1, 1)


def detect_chop(hist: Sequence[Any], snap: Any, recent_signals: Sequence[str], params: ChopParams) -> ChopState:
    closes = [bar.close for bar in hist]
    fast = mean(closes[-20:])
    slow = mean(closes[-60:])
    adx = calculate_adx(hist[-120:], 14)
    efficiency = range_efficiency(hist, 48)
    ratio = atr_ratio(hist, 14, 96)
    slope_atr = abs(fast - slow) / max(snap.atr, POINT)
    alternation = alternating_signal_rate(recent_signals[-36:])

    points = 0
    reasons: List[str] = []
    if adx <= params.adx_max:
        points += 1
        reasons.append("adx")
    if efficiency <= params.efficiency_max:
        points += 1
        reasons.append("efficiency")
    if ratio <= params.atr_ratio_max:
        points += 1
        reasons.append("atr_compression")
    if slope_atr <= params.slope_atr_max:
        points += 1
        reasons.append("flat_sma")
    if alternation >= params.alternation_min:
        points += 1
        reasons.append("alternating_signals")
    if abs(float(snap.score)) <= params.min_score:
        points += 1
        reasons.append("weak_score")

    is_chop = points >= params.min_chop_points
    return ChopState(
        is_chop=is_chop,
        score=float(points),
        adx=adx,
        efficiency=efficiency,
        atr_ratio=ratio,
        slope_atr=slope_atr,
        alternation=alternation,
        reason="+".join(reasons) if reasons else "none",
    )


def false_breakout_signal(hist: Sequence[Any], atr: float, lookback: int = 20) -> str:
    if len(hist) < lookback + 1:
        return "NONE"
    last = hist[-1]
    prior = hist[-lookback - 1:-1]
    prior_high = max(bar.high for bar in prior)
    prior_low = min(bar.low for bar in prior)
    bar_range = max(last.high - last.low, POINT)
    body_high = max(last.open, last.close)
    body_low = min(last.open, last.close)
    upper_wick_ratio = (last.high - body_high) / bar_range
    lower_wick_ratio = (body_low - last.low) / bar_range
    upthrust = last.high >= prior_high + 0.12 * atr and last.close <= prior_high - 0.03 * atr and upper_wick_ratio >= 0.35
    spring = last.low <= prior_low - 0.12 * atr and last.close >= prior_low + 0.03 * atr and lower_wick_ratio >= 0.35
    if upthrust:
        return "SELL"
    if spring:
        return "BUY"
    return "NONE"


def close_if_needed(pos: OpenPosition, i: int, nxt: Any, max_hold_bars: int) -> Optional[Tuple[Trade, float]]:
    exit_price: Optional[float] = None
    reason = ""
    if pos.side == "BUY":
        if nxt.low <= pos.sl:
            exit_price, reason = pos.sl, "SL"
        elif nxt.high >= pos.tp:
            exit_price, reason = pos.tp, "TP"
    else:
        if nxt.high >= pos.sl:
            exit_price, reason = pos.sl, "SL"
        elif nxt.low <= pos.tp:
            exit_price, reason = pos.tp, "TP"
    bars_held = i - pos.entry_i
    if exit_price is None and bars_held >= max_hold_bars:
        exit_price, reason = nxt.close, "TIME"
    if exit_price is None:
        return None
    gross, net, commission, spread_cost = costs(pos.entry, exit_price, pos.volume, pos.side)
    risk_amount = abs(pos.entry - pos.orig_sl) * pos.volume * CONTRACT_SIZE
    return Trade(
        strategy="chop_regime_hedge_helper",
        signal_source=pos.sleeve,
        entry_time=pos.entry_time.isoformat(),
        exit_time=nxt.time.isoformat(),
        side=pos.side,
        entry=pos.entry,
        exit=exit_price,
        sl=pos.sl,
        tp=pos.tp,
        volume=pos.volume,
        gross_pnl=gross,
        net_pnl=net,
        commission=commission,
        spread_cost=spread_cost,
        r_multiple=gross / max(risk_amount, 1e-9),
        reason=reason,
        bars_held=bars_held,
        atr=pos.atr,
        score=pos.score,
        htf_signal=pos.htf_signal,
        session=pos.session,
    ), net


def summarize(trades: Sequence[Trade], initial_equity: float = INITIAL_EQUITY) -> Dict[str, Any]:
    equity = initial_equity
    peak = equity
    max_dd = 0.0
    wins: List[Trade] = []
    losses: List[Trade] = []
    for trade in sorted(trades, key=lambda t: t.exit_time):
        equity += trade.net_pnl
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / max(peak, 1e-9) - 1.0)
        if trade.net_pnl > 0:
            wins.append(trade)
        else:
            losses.append(trade)
    gp = sum(t.net_pnl for t in wins)
    gl = -sum(t.net_pnl for t in losses)
    returns = [t.net_pnl / initial_equity for t in trades]
    return {
        "trades": len(trades),
        "net_pnl": round(equity - initial_equity, 2),
        "return_pct": round((equity / initial_equity - 1.0) * 100.0, 2),
        "win_rate_pct": round(100.0 * len(wins) / len(trades), 2) if trades else 0.0,
        "profit_factor": round(gp / gl, 3) if gl > 0 else (999.0 if gp > 0 else 0.0),
        "max_dd_pct": round(max_dd * 100.0, 2),
        "expectancy_usd": round(mean([t.net_pnl for t in trades]), 2) if trades else 0.0,
        "avg_r": round(mean([t.r_multiple for t in trades]), 3) if trades else 0.0,
        "approx_sharpe_per_trade": round(mean(returns) / max(stdev(returns), 1e-12) * math.sqrt(len(returns)), 3) if len(returns) > 1 else 0.0,
    }


def week_key(value: str) -> str:
    iso = dt.datetime.fromisoformat(value).isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def weekly_stats(trades: Sequence[Trade]) -> Dict[str, Any]:
    grouped: Dict[str, float] = {}
    for trade in trades:
        key = week_key(trade.entry_time)
        grouped[key] = grouped.get(key, 0.0) + trade.net_pnl
    values = list(grouped.values())
    if not values:
        return {"active_weeks": 0, "positive_active_week_pct": 0.0, "avg_weekly_return_pct": 0.0, "worst_week_pct": 0.0, "best_week_pct": 0.0}
    return {
        "active_weeks": len(values),
        "positive_active_week_pct": round(100.0 * sum(1 for v in values if v > 0) / len(values), 2),
        "avg_weekly_return_pct": round(statistics.fmean(values) / INITIAL_EQUITY * 100.0, 3),
        "worst_week_pct": round(min(values) / INITIAL_EQUITY * 100.0, 3),
        "best_week_pct": round(max(values) / INITIAL_EQUITY * 100.0, 3),
    }


def open_position(
    sleeve: str,
    side: str,
    i: int,
    nxt: Any,
    snap: Any,
    equity: float,
    risk_pct: float,
    max_lots: float,
    reward_multiple: float,
    context: str,
) -> Optional[OpenPosition]:
    entry = nxt.open
    sl_dist = snap.atr * Params(name="tmp", mode="trend_only").stop_atr
    tp_dist = sl_dist * reward_multiple
    sl = entry - sl_dist if side == "BUY" else entry + sl_dist
    tp = entry + tp_dist if side == "BUY" else entry - tp_dist
    vol = sized_volume(equity, risk_pct, max_lots, entry, sl)
    if vol < 0.01:
        return None
    return OpenPosition(
        sleeve=sleeve,
        side=side,
        entry=entry,
        sl=sl,
        orig_sl=sl,
        tp=tp,
        volume=vol,
        entry_time=nxt.time,
        entry_i=i + 1,
        atr=snap.atr,
        score=snap.score,
        htf_signal=snap.compensated_htf_signal,
        session=snap.session,
        context=context,
    )


def simulate(m5_bars: Sequence[Any], h1_bars: Sequence[Any], m15_bars: Sequence[Any], cp: ChopParams) -> Dict[str, Any]:
    base = Params(name="trend_with_chop_regime_helper", mode="trend_only")
    h1_times = [bar.time for bar in h1_bars]
    m15_times = [bar.time for bar in m15_bars]
    h1_cache: Dict[int, Tuple[float, float, str]] = {}
    warmup = max(300, base.slow_sma + base.breakout_lookback + base.atr_period + 100)

    trend_pos: Optional[OpenPosition] = None
    helper_pos: Optional[OpenPosition] = None
    cooldown = {"trend": -1, "helper": -1}
    equity = INITIAL_EQUITY
    trend_trades: List[Trade] = []
    helper_trades: List[Trade] = []
    combined_trades: List[Trade] = []
    recent_signals: deque[str] = deque(maxlen=48)
    signal_counts = Counter()
    chop_reason_counts = Counter()

    for i in range(warmup, len(m5_bars) - 1):
        hist = m5_bars[max(0, i - 320):i + 1]
        snap = build_snapshot(hist, h1_bars, h1_times, h1_cache, m15_bars, m15_times, base)
        if snap is None:
            continue
        nxt = m5_bars[i + 1]
        recent_signals.append(snap.primary_signal)
        chop = detect_chop(hist, snap, list(recent_signals), cp)
        if chop.is_chop:
            signal_counts["chop_bars"] += 1
            chop_reason_counts[chop.reason] += 1
        signal_counts[f"primary_{snap.primary_signal}"] += 1

        if trend_pos is not None:
            closed = close_if_needed(trend_pos, i, nxt, base.max_hold_bars)
            if closed is not None:
                trade, net = closed
                trade = dataclasses.replace(trade, strategy=cp.name, signal_source="trend")
                trend_trades.append(trade)
                combined_trades.append(trade)
                equity += net
                cooldown["trend"] = i + base.cooldown_bars
                trend_pos = None

        if helper_pos is not None:
            closed = close_if_needed(helper_pos, i, nxt, cp.max_hold_bars)
            if closed is not None:
                trade, net = closed
                trade = dataclasses.replace(trade, strategy=cp.name, signal_source="helper")
                helper_trades.append(trade)
                combined_trades.append(trade)
                equity += net
                cooldown["helper"] = i + base.cooldown_bars
                helper_pos = None

        primary = snap.primary_signal
        trend_allowed = True
        trend_risk_mult = 1.0
        if chop.is_chop and cp.action == "pause_trend":
            trend_allowed = False
            signal_counts["trend_paused_in_chop"] += 1
        elif chop.is_chop and cp.action == "scale_trend":
            trend_risk_mult = cp.risk_mult
            signal_counts["trend_scaled_in_chop"] += 1
        elif chop.is_chop and cp.action == "session_aware_gate":
            if snap.session == "asia":
                trend_allowed = False
                signal_counts["trend_paused_in_chop"] += 1
                signal_counts["trend_paused_asia_chop"] += 1
            else:
                trend_risk_mult = cp.risk_mult
                signal_counts["trend_scaled_in_chop"] += 1
                signal_counts[f"trend_scaled_{snap.session}_chop"] += 1

        if trend_pos is None and trend_allowed and i >= cooldown["trend"] and primary in {"BUY", "SELL"}:
            pos = open_position(
                "trend",
                primary,
                i,
                nxt,
                snap,
                equity,
                base.risk_pct * trend_risk_mult,
                base.max_lots * trend_risk_mult,
                base.reward_multiple,
                f"chop={chop.is_chop};{chop.reason}",
            )
            if pos is not None:
                trend_pos = pos
                signal_counts["opened_trend"] += 1

        helper_signal = "NONE"
        if chop.is_chop:
            if cp.action == "hedge_opposite" and primary in {"BUY", "SELL"}:
                helper_signal = "SELL" if primary == "BUY" else "BUY"
            elif cp.action == "fade_breakout":
                helper_signal = false_breakout_signal(hist, snap.atr)
                if cp.require_primary_signal and primary not in {"BUY", "SELL"}:
                    helper_signal = "NONE"

        if helper_pos is None and helper_signal in {"BUY", "SELL"} and i >= cooldown["helper"]:
            pos = open_position(
                "helper",
                helper_signal,
                i,
                nxt,
                snap,
                equity,
                base.risk_pct * cp.risk_mult,
                base.max_lots * cp.risk_mult,
                cp.reward_multiple,
                f"chop_score={chop.score};{chop.reason};primary={primary};adx={chop.adx:.1f};eff={chop.efficiency:.2f}",
            )
            if pos is not None:
                helper_pos = pos
                signal_counts["opened_helper"] += 1

    combined_trades.sort(key=lambda t: t.exit_time)
    return {
        "params": dataclasses.asdict(cp),
        "trend": summarize(trend_trades),
        "helper": summarize(helper_trades),
        "combined": summarize(combined_trades),
        "trend_weekly": weekly_stats(trend_trades),
        "helper_weekly": weekly_stats(helper_trades),
        "combined_weekly": weekly_stats(combined_trades),
        "helper_reason_counts": dict(Counter(t.reason for t in helper_trades)),
        "helper_side_counts": dict(Counter(t.side for t in helper_trades)),
        "signal_counts": dict(signal_counts),
        "chop_reason_counts": dict(chop_reason_counts.most_common(20)),
        "trend_trades": trend_trades,
        "helper_trades": helper_trades,
        "combined_trades": combined_trades,
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row.keys())) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if fields:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m5 = load_m5_csv(CSV_PATH)
    h1 = aggregate_bars(m5, 60)
    m15 = aggregate_bars(m5, 15)

    baseline_params = ChopParams(
        "baseline_trend_only_no_chop_action", "none", 0.0, 0.0, 0.0, 0.0, 999.0, 0.0, 0.0, 1.0, 48, False
    )
    baseline_result = simulate(m5, h1, m15, baseline_params)
    baseline_combined = baseline_result["combined"]

    grid: List[ChopParams] = [
        ChopParams("chop_hedge_balanced_r0.25_rr1.5", "hedge_opposite", 22.0, 0.22, 0.92, 1.25, 0.45, 0.75, 0.25, 1.5, 48),
        ChopParams("chop_hedge_balanced_r0.50_rr1.5", "hedge_opposite", 22.0, 0.22, 0.92, 1.25, 0.45, 0.75, 0.50, 1.5, 48),
        ChopParams("chop_hedge_strict_r0.25_rr2.0", "hedge_opposite", 18.0, 0.18, 0.85, 1.00, 0.55, 0.65, 0.25, 2.0, 48),
        ChopParams("chop_hedge_strict_r0.50_rr2.0", "hedge_opposite", 18.0, 0.18, 0.85, 1.00, 0.55, 0.65, 0.50, 2.0, 48),
        ChopParams("chop_fade_balanced_r0.25_rr1.5", "fade_breakout", 22.0, 0.22, 0.92, 1.25, 0.45, 0.75, 0.25, 1.5, 48),
        ChopParams("chop_fade_balanced_r0.50_rr1.5", "fade_breakout", 22.0, 0.22, 0.92, 1.25, 0.45, 0.75, 0.50, 1.5, 48),
        ChopParams("chop_fade_strict_r0.25_rr2.0", "fade_breakout", 18.0, 0.18, 0.85, 1.00, 0.55, 0.65, 0.25, 2.0, 48),
        ChopParams("chop_fade_strict_r0.50_rr2.0", "fade_breakout", 18.0, 0.18, 0.85, 1.00, 0.55, 0.65, 0.50, 2.0, 48),
        ChopParams("chop_pause_balanced", "pause_trend", 22.0, 0.22, 0.92, 1.25, 0.45, 0.75, 0.0, 1.5, 48, False),
        ChopParams("chop_pause_strict", "pause_trend", 18.0, 0.18, 0.85, 1.00, 0.55, 0.65, 0.0, 2.0, 48, False),
        ChopParams("chop_pause_strict_p4", "pause_trend", 18.0, 0.18, 0.85, 1.00, 0.55, 0.65, 0.0, 2.0, 48, False, 4),
        ChopParams("chop_pause_adx20_p4", "pause_trend", 20.0, 0.20, 0.88, 1.10, 0.50, 0.70, 0.0, 2.0, 48, False, 4),
        ChopParams("chop_pause_flat_p4", "pause_trend", 22.0, 0.18, 0.90, 0.85, 0.50, 0.70, 0.0, 2.0, 48, False, 4),
        ChopParams("chop_scale_balanced_r0.50", "scale_trend", 22.0, 0.22, 0.92, 1.25, 0.45, 0.75, 0.50, 1.5, 48, False),
        ChopParams("chop_scale_strict_r0.50", "scale_trend", 18.0, 0.18, 0.85, 1.00, 0.55, 0.65, 0.50, 2.0, 48, False),
        ChopParams("chop_scale_strict_p4_r0.50", "scale_trend", 18.0, 0.18, 0.85, 1.00, 0.55, 0.65, 0.50, 2.0, 48, False, 4),
        ChopParams("chop_scale_strict_p4_r0.25", "scale_trend", 18.0, 0.18, 0.85, 1.00, 0.55, 0.65, 0.25, 2.0, 48, False, 4),
        ChopParams("chop_session_gate_strict_r0.50", "session_aware_gate", 18.0, 0.18, 0.85, 1.00, 0.55, 0.65, 0.50, 2.0, 48, False),
        ChopParams("chop_session_gate_strict_r0.25", "session_aware_gate", 18.0, 0.18, 0.85, 1.00, 0.55, 0.65, 0.25, 2.0, 48, False),
        ChopParams("chop_session_gate_balanced_r0.50", "session_aware_gate", 22.0, 0.22, 0.92, 1.25, 0.45, 0.75, 0.50, 1.5, 48, False),
    ]

    rows: List[Dict[str, Any]] = []
    result_by_name: Dict[str, Dict[str, Any]] = {}
    for cp in grid:
        result = simulate(m5, h1, m15, cp)
        result_by_name[cp.name] = result
        row = {
            **result["params"],
            **{f"trend_{k}": v for k, v in result["trend"].items()},
            **{f"helper_{k}": v for k, v in result["helper"].items()},
            **{f"combined_{k}": v for k, v in result["combined"].items()},
            **{f"combined_weekly_{k}": v for k, v in result["combined_weekly"].items()},
            "chop_bars": result["signal_counts"].get("chop_bars", 0),
            "opened_helper": result["signal_counts"].get("opened_helper", 0),
            "trend_paused_in_chop": result["signal_counts"].get("trend_paused_in_chop", 0),
            "trend_scaled_in_chop": result["signal_counts"].get("trend_scaled_in_chop", 0),
            "baseline_return_pct": baseline_combined["return_pct"],
            "baseline_max_dd_pct": baseline_combined["max_dd_pct"],
            "baseline_profit_factor": baseline_combined["profit_factor"],
        }
        row["dd_improvement_vs_baseline_pct"] = round(row["combined_max_dd_pct"] - baseline_combined["max_dd_pct"], 2)
        row["return_delta_vs_baseline_pct"] = round(row["combined_return_pct"] - baseline_combined["return_pct"], 2)
        rows.append(row)

    rows.sort(
        key=lambda r: (
            float(r["dd_improvement_vs_baseline_pct"]),
            float(r["return_delta_vs_baseline_pct"]),
            float(r["combined_profit_factor"]),
            float(r["combined_return_pct"]),
        ),
        reverse=True,
    )
    write_csv(OUT_DIR / "chop_regime_hedge_grid.csv", rows)

    top_result: Optional[Dict[str, Any]] = result_by_name[str(rows[0]["name"])] if rows else None
    if top_result is not None:
        write_csv(OUT_DIR / "best_chop_helper_trades.csv", [dataclasses.asdict(t) for t in top_result["helper_trades"]])
        write_csv(OUT_DIR / "best_chop_combined_trades.csv", [dataclasses.asdict(t) for t in top_result["combined_trades"]])

    diagnostics = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "initial_equity": INITIAL_EQUITY,
        "source_csv": str(CSV_PATH),
        "m5_bars": len(m5),
        "m5_start": m5[0].time.isoformat(),
        "m5_end": m5[-1].time.isoformat(),
        "baseline": baseline_combined,
        "grid_count": len(grid),
        "top_rows": rows[:10],
        "top_chop_reason_counts": top_result["chop_reason_counts"] if top_result else {},
        "top_signal_counts": top_result["signal_counts"] if top_result else {},
        "outputs": {
            "grid": str(OUT_DIR / "chop_regime_hedge_grid.csv"),
            "best_helper_trades": str(OUT_DIR / "best_chop_helper_trades.csv"),
            "best_combined_trades": str(OUT_DIR / "best_chop_combined_trades.csv"),
        },
    }
    (OUT_DIR / "chop_regime_hedge_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
