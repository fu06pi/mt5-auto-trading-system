#!/usr/bin/env python3.14
"""Weekly breakdown for XAUUSD complement and current+complement variants.

Read-only research script. No MT5 orders and no active_plan edits.
"""
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from research_xauusd_trend_plus_complement_backtest import (
    INITIAL_EQUITY,
    Params,
    Trade,
    backtest,
    backtest_parallel_sleeves,
    mean,
    stdev,
)
from research_xauusd_trend_plus_complement_long_cache_backtest import (
    CSV_PATH,
    aggregate_bars,
    load_m5_csv,
)

OUT_DIR = Path("/home/chain4655/Documents/Projects/MT5/backtest_reports_trend_plus_complement/weekly")


def week_key(ts: dt.datetime) -> str:
    iso = ts.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def parse_trade_time(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value)


def weekly_keys_from_bars(m5_bars: Sequence[Any]) -> List[str]:
    keys = []
    seen = set()
    for bar in m5_bars:
        key = week_key(bar.time)
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def summarize_trades(trades: Sequence[Trade]) -> Dict[str, Any]:
    wins = [trade for trade in trades if trade.net_pnl > 0]
    losses = [trade for trade in trades if trade.net_pnl <= 0]
    gp = sum(trade.net_pnl for trade in wins)
    gl = -sum(trade.net_pnl for trade in losses)
    returns = [trade.net_pnl / INITIAL_EQUITY for trade in trades]
    net = sum(trade.net_pnl for trade in trades)
    equity = INITIAL_EQUITY
    peak = equity
    max_dd = 0.0
    for trade in trades:
        equity += trade.net_pnl
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / max(peak, 1e-9) - 1.0)
    return {
        "trades": len(trades),
        "net_pnl": round(net, 2),
        "return_pct": round(net / INITIAL_EQUITY * 100.0, 2),
        "win_rate_pct": round(100.0 * len(wins) / len(trades), 2) if trades else 0.0,
        "profit_factor": round(gp / gl, 3) if gl > 0 else (999.0 if gp > 0 else 0.0),
        "max_dd_pct": round(max_dd * 100.0, 2),
        "expectancy_usd": round(mean([trade.net_pnl for trade in trades]), 2) if trades else 0.0,
        "avg_r": round(mean([trade.r_multiple for trade in trades]), 3) if trades else 0.0,
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


def weekly_rows(strategy: str, trades: Sequence[Trade], all_weeks: Sequence[str]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Trade]] = defaultdict(list)
    for trade in trades:
        grouped[week_key(parse_trade_time(trade.entry_time))].append(trade)
    rows = []
    for key in all_weeks:
        rows.append({"strategy": strategy, "week": key, **summarize_trades(grouped.get(key, []))})
    return rows


def strategy_week_summary(strategy: str, rows: Sequence[Dict[str, Any]], full_metrics: Dict[str, Any]) -> Dict[str, Any]:
    active = [row for row in rows if int(row["trades"]) > 0]
    positive = [row for row in rows if float(row["net_pnl"]) > 0]
    negative = [row for row in rows if float(row["net_pnl"]) < 0]
    worst = min(rows, key=lambda row: float(row["net_pnl"])) if rows else {}
    best = max(rows, key=lambda row: float(row["net_pnl"])) if rows else {}
    return {
        "strategy": strategy,
        "weeks_total": len(rows),
        "active_weeks": len(active),
        "positive_weeks": len(positive),
        "negative_weeks": len(negative),
        "flat_weeks": len(rows) - len(positive) - len(negative),
        "positive_active_week_pct": round(100.0 * len([r for r in active if float(r["net_pnl"]) > 0]) / len(active), 2) if active else 0.0,
        "weekly_avg_pnl": round(mean([float(row["net_pnl"]) for row in rows]), 2) if rows else 0.0,
        "active_weekly_avg_pnl": round(mean([float(row["net_pnl"]) for row in active]), 2) if active else 0.0,
        "worst_week": worst.get("week"),
        "worst_week_pnl": worst.get("net_pnl"),
        "best_week": best.get("week"),
        "best_week_pnl": best.get("net_pnl"),
        **{f"full_{key}": value for key, value in full_metrics.items() if key in {"trades", "net_pnl", "return_pct", "win_rate_pct", "profit_factor", "max_dd_pct", "avg_r", "approx_sharpe_per_trade"}},
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m5_bars = load_m5_csv(CSV_PATH)
    h1_bars = aggregate_bars(m5_bars, 60)
    m15_bars = aggregate_bars(m5_bars, 15)
    all_weeks = weekly_keys_from_bars(m5_bars)

    variants = [
        Params(name="current_trend_only_weekly", mode="trend_only"),
        Params(name="complement_fb_sell_only_weekly", mode="complement_only"),
        Params(name="current_trend_plus_complement_overlay_weekly", mode="trend_plus_complement"),
        Params(name="parallel_independent_sleeves_weekly", mode="parallel_sleeves"),
    ]

    all_weekly_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    full_rows: List[Dict[str, Any]] = []
    all_trades: List[Dict[str, Any]] = []

    for params in variants:
        if params.mode == "parallel_sleeves":
            trades, metrics = backtest_parallel_sleeves(m5_bars, h1_bars, m15_bars, params)
        else:
            trades, metrics = backtest(m5_bars, h1_bars, m15_bars, params)
        rows = weekly_rows(params.name, trades, all_weeks)
        all_weekly_rows.extend(rows)
        summary_rows.append(strategy_week_summary(params.name, rows, metrics))
        full_rows.append({"strategy": params.name, **metrics})
        for trade in trades:
            row = dataclasses.asdict(trade)
            row["week"] = week_key(parse_trade_time(trade.entry_time))
            all_trades.append(row)

    write_csv(OUT_DIR / "weekly_strategy_rows.csv", all_weekly_rows)
    write_csv(OUT_DIR / "weekly_strategy_summary.csv", summary_rows)
    write_csv(OUT_DIR / "weekly_full_metrics.csv", full_rows)
    write_csv(OUT_DIR / "weekly_all_trades.csv", all_trades)

    diagnostics = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source_csv": str(CSV_PATH),
        "m5_bars": len(m5_bars),
        "m5_start": m5_bars[0].time.isoformat(),
        "m5_end": m5_bars[-1].time.isoformat(),
        "weeks_total": len(all_weeks),
        "note": "Weekly rows are grouped by ISO week from full continuous backtests; equity is not reset each week.",
        "outputs": {
            "weekly_rows": str(OUT_DIR / "weekly_strategy_rows.csv"),
            "summary": str(OUT_DIR / "weekly_strategy_summary.csv"),
            "full_metrics": str(OUT_DIR / "weekly_full_metrics.csv"),
            "trades": str(OUT_DIR / "weekly_all_trades.csv"),
        },
    }
    with (OUT_DIR / "weekly_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    print(json.dumps({"summary": summary_rows, "full_metrics": full_rows, "diagnostics": diagnostics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
