#!/usr/bin/env python3.14
"""Stability tests for XAUUSD complement false-breakout SELL-only strategy.

Read-only research script. No MT5 orders and no active_plan edits.
"""
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from research_xauusd_trend_plus_complement_backtest import (
    COMMISSION_PER_LOT,
    INITIAL_EQUITY,
    POINT,
    SPREAD_POINTS,
    Params,
    Trade,
    backtest,
    mean,
    stdev,
)
from research_xauusd_trend_plus_complement_long_cache_backtest import (
    CSV_PATH,
    aggregate_bars,
    load_m5_csv,
)

OUT_DIR = Path("/home/chain4655/Documents/Projects/MT5/backtest_reports_trend_plus_complement/stability")


def adjusted_net(trade: Trade, spread_points: float, commission_per_lot: float) -> float:
    spread_cost = spread_points * POINT * trade.volume * 100.0
    commission = commission_per_lot * trade.volume
    return trade.gross_pnl - spread_cost - commission


def summarize_adjusted(
    trades: Sequence[Trade], spread_points: float = SPREAD_POINTS, commission_per_lot: float = COMMISSION_PER_LOT
) -> Dict[str, Any]:
    equity = INITIAL_EQUITY
    peak = equity
    max_dd = 0.0
    nets: List[float] = []
    gp = 0.0
    gl = 0.0
    wins = 0
    for trade in trades:
        net = adjusted_net(trade, spread_points, commission_per_lot)
        nets.append(net)
        equity += net
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / max(peak, 1e-9) - 1.0)
        if net > 0:
            wins += 1
            gp += net
        else:
            gl -= net
    returns = [net / INITIAL_EQUITY for net in nets]
    return {
        "trades": len(trades),
        "net_pnl": round(sum(nets), 2),
        "return_pct": round(sum(nets) / INITIAL_EQUITY * 100.0, 2),
        "win_rate_pct": round(100.0 * wins / len(trades), 2) if trades else 0.0,
        "profit_factor": round(gp / gl, 3) if gl > 0 else (999.0 if gp > 0 else 0.0),
        "max_dd_pct": round(max_dd * 100.0, 2),
        "expectancy_usd": round(mean(nets), 2) if nets else 0.0,
        "avg_r_gross": round(mean([trade.r_multiple for trade in trades]), 3) if trades else 0.0,
        "approx_sharpe_per_trade": round(mean(returns) / max(stdev(returns), 1e-12) * math.sqrt(len(returns)), 3) if len(returns) > 1 else 0.0,
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row.keys())) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if fields:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_complement(m5_bars: Sequence[Any], name: str, params: Params) -> tuple[List[Trade], Dict[str, Any]]:
    h1_bars = aggregate_bars(list(m5_bars), 60)
    m15_bars = aggregate_bars(list(m5_bars), 15)
    return backtest(m5_bars, h1_bars, m15_bars, dataclasses.replace(params, name=name, mode="complement_only"))


def segment_bars(m5_bars: Sequence[Any], months: int) -> List[tuple[str, List[Any]]]:
    if not m5_bars:
        return []
    start = m5_bars[0].time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end = m5_bars[-1].time
    segments: List[tuple[str, List[Any]]] = []
    cur = start
    while cur < end:
        year = cur.year + (cur.month - 1 + months) // 12
        month = (cur.month - 1 + months) % 12 + 1
        nxt = cur.replace(year=year, month=month)
        rows = [bar for bar in m5_bars if cur <= bar.time < nxt]
        if rows:
            segments.append((f"{cur:%Y-%m}_to_{nxt:%Y-%m}", rows))
        cur = nxt
    return segments


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m5_bars = load_m5_csv(CSV_PATH)
    base = Params(name="complement_base", mode="complement_only")

    base_trades, base_metrics = run_complement(m5_bars, "complement_base", base)

    # Trade-outcome consistency by calendar month from the full run.
    by_month: Dict[str, List[Trade]] = defaultdict(list)
    for trade in base_trades:
        by_month[trade.entry_time[:7]].append(trade)
    month_rows = []
    for month, trades in sorted(by_month.items()):
        month_rows.append({"month": month, **summarize_adjusted(trades)})

    # Independent segment re-runs, reset equity each segment.
    segment_rows = []
    for span_months in (3, 6):
        for label, rows in segment_bars(m5_bars, span_months):
            trades, metrics = run_complement(rows, f"segment_{span_months}m_{label}", base)
            segment_rows.append({"span_months": span_months, "segment": label, "bars": len(rows), **metrics})

    # Parameter sensitivity around the selected setting. Keep grid compact to avoid overfitting theatre.
    grid_rows = []
    for fb_lookback in (15, 20, 25):
        for fb_min_atr in (0.10, 0.15, 0.20):
            for fb_wick_ratio in (0.35, 0.45, 0.55):
                params = dataclasses.replace(
                    base,
                    name=f"grid_lb{fb_lookback}_min{fb_min_atr}_wick{fb_wick_ratio}",
                    fb_lookback=fb_lookback,
                    fb_min_atr=fb_min_atr,
                    fb_wick_ratio=fb_wick_ratio,
                )
                trades, metrics = run_complement(m5_bars, params.name, params)
                grid_rows.append({
                    "fb_lookback": fb_lookback,
                    "fb_min_atr": fb_min_atr,
                    "fb_wick_ratio": fb_wick_ratio,
                    **metrics,
                })

    # Cost sensitivity using same base trades.
    cost_rows = []
    for spread_points in (20, 40, 60, 80, 120):
        cost_rows.append({
            "spread_points": spread_points,
            "commission_per_lot": COMMISSION_PER_LOT,
            **summarize_adjusted(base_trades, spread_points=spread_points, commission_per_lot=COMMISSION_PER_LOT),
        })
    for commission in (7, 10, 14):
        cost_rows.append({
            "spread_points": SPREAD_POINTS,
            "commission_per_lot": commission,
            **summarize_adjusted(base_trades, spread_points=SPREAD_POINTS, commission_per_lot=commission),
        })

    session_rows = []
    by_session: Dict[str, List[Trade]] = defaultdict(list)
    for trade in base_trades:
        by_session[trade.session].append(trade)
    for session, trades in sorted(by_session.items()):
        session_rows.append({"session": session, **summarize_adjusted(trades)})

    reason_rows = []
    by_reason: Dict[str, List[Trade]] = defaultdict(list)
    for trade in base_trades:
        by_reason[trade.reason].append(trade)
    for reason, trades in sorted(by_reason.items()):
        reason_rows.append({"reason": reason, **summarize_adjusted(trades)})

    diagnostics = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source_csv": str(CSV_PATH),
        "bars": len(m5_bars),
        "start": m5_bars[0].time.isoformat(),
        "end": m5_bars[-1].time.isoformat(),
        "base_params": dataclasses.asdict(base),
        "base_metrics": base_metrics,
        "month_positive_count": sum(1 for row in month_rows if row["net_pnl"] > 0),
        "month_count": len(month_rows),
        "segment_positive_count": sum(1 for row in segment_rows if row["net_pnl"] > 0),
        "segment_count": len(segment_rows),
        "grid_profitable_count": sum(1 for row in grid_rows if row["net_pnl"] > 0),
        "grid_pf_gt_1_count": sum(1 for row in grid_rows if row["profit_factor"] > 1.0),
        "signal_counts": dict(Counter(trade.side for trade in base_trades)),
        "outputs": {
            "base_trades": str(OUT_DIR / "complement_stability_base_trades.csv"),
            "months": str(OUT_DIR / "complement_stability_months.csv"),
            "segments": str(OUT_DIR / "complement_stability_segments.csv"),
            "grid": str(OUT_DIR / "complement_stability_param_grid.csv"),
            "costs": str(OUT_DIR / "complement_stability_costs.csv"),
        },
    }

    write_csv(OUT_DIR / "complement_stability_base_trades.csv", [dataclasses.asdict(t) for t in base_trades])
    write_csv(OUT_DIR / "complement_stability_months.csv", month_rows)
    write_csv(OUT_DIR / "complement_stability_segments.csv", segment_rows)
    write_csv(OUT_DIR / "complement_stability_param_grid.csv", grid_rows)
    write_csv(OUT_DIR / "complement_stability_costs.csv", cost_rows)
    write_csv(OUT_DIR / "complement_stability_sessions.csv", session_rows)
    write_csv(OUT_DIR / "complement_stability_reasons.csv", reason_rows)
    with (OUT_DIR / "complement_stability_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "base_metrics": base_metrics,
        "month_rows": month_rows,
        "segment_rows": segment_rows,
        "cost_rows": cost_rows,
        "session_rows": session_rows,
        "reason_rows": reason_rows,
        "grid_summary": {
            "rows": len(grid_rows),
            "profitable": diagnostics["grid_profitable_count"],
            "pf_gt_1": diagnostics["grid_pf_gt_1_count"],
            "best": sorted(grid_rows, key=lambda row: row["net_pnl"], reverse=True)[:5],
            "worst": sorted(grid_rows, key=lambda row: row["net_pnl"])[:5],
        },
        "out_dir": str(OUT_DIR),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
