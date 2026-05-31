#!/usr/bin/env python3.14
"""Backtest active Doomsday V4 parameters on the most recent 90 calendar days of XAUUSD M5 data.

Uses the research_backtest_sampling_doomsday_v4 engine and writes CSV/JSON artifacts.
"""
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
from collections import Counter
from pathlib import Path

from research_backtest_sampling_doomsday_v4 import (
    MAX_H1_BARS,
    MAX_M5_BARS,
    OUT_DIR as BASE_OUT_DIR,
    Params,
    add_inference_flags,
    backtest,
    fetch_bars,
    grouped_metrics,
    representativeness,
)

OUT_DIR = Path("/home/chain4655/Documents/Projects/MT5/backtest_reports_doomsday_last_3m")
DAYS = 90


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m5_all = fetch_bars("M5", MAX_M5_BARS)
    h1_all = fetch_bars("H1", MAX_H1_BARS)
    if not m5_all:
        raise RuntimeError("No M5 bars fetched")
    end = m5_all[-1].time
    start = end - dt.timedelta(days=DAYS)
    m5 = [bar for bar in m5_all if start <= bar.time <= end]
    h1 = [bar for bar in h1_all if start <= bar.time <= end]
    if len(m5) < 500:
        raise RuntimeError(f"Too few M5 bars for 3M backtest: {len(m5)}")

    variants = [
        Params(name="active_sprint_v4"),
        Params(
            name="stricter_hv_gate",
            high_vol_atr_pct=0.0018,
            high_vol_min_breakout_atr=0.25,
            high_vol_min_close_location=0.68,
            high_vol_spike_atr=2.8,
        ),
        Params(
            name="looser_hv_gate",
            high_vol_atr_pct=0.0012,
            high_vol_min_momentum=0.55,
            high_vol_min_breakout_atr=0.10,
            high_vol_min_close_location=0.55,
        ),
        Params(name="active_rr_1p6", reward_multiple=1.6),
        Params(name="active_rr_2p8", reward_multiple=2.8),
        Params(name="no_high_vol_filter", high_vol_only=False),
    ]

    summary_rows: list[dict] = []
    all_trades = []
    diagnostics = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "period_mode": "last_90_days_from_latest_m5_bar",
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "m5_bars": len(m5),
        "h1_bars_in_period": len(h1),
        "representativeness_vs_available_h1": representativeness(m5, h1_all),
        "strategy_grouped": {},
    }

    for params in variants:
        trades, metrics, snapshots = backtest(m5, params, label="last_3m")
        metrics = add_inference_flags(metrics, trades)
        row = {"strategy": params.name, **metrics}
        summary_rows.append(row)
        all_trades.extend(trades)
        diagnostics["strategy_grouped"][params.name] = {
            "fold": grouped_metrics(trades, "fold"),
            "session": grouped_metrics(trades, "session"),
            "vol_bucket": grouped_metrics(trades, "vol_bucket"),
            "months_with_trades": len({trade.entry_time[:7] for trade in trades}),
            "trade_month_counts": dict(Counter(trade.entry_time[:7] for trade in trades)),
        }

    summary_fields = list(summary_rows[0].keys()) if summary_rows else []
    trade_fields = [field.name for field in dataclasses.fields(all_trades[0])] if all_trades else []
    write_csv(OUT_DIR / "doomsday_v4_last_3m_summary.csv", summary_rows, summary_fields)
    if all_trades:
        write_csv(
            OUT_DIR / "doomsday_v4_last_3m_trades.csv",
            [dataclasses.asdict(trade) for trade in all_trades],
            trade_fields,
        )
    else:
        (OUT_DIR / "doomsday_v4_last_3m_trades.csv").write_text("", encoding="utf-8")
    with (OUT_DIR / "doomsday_v4_last_3m_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    print(json.dumps({"summary": summary_rows, "diagnostics": diagnostics, "out_dir": str(OUT_DIR)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
