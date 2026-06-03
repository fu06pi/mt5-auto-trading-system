#!/usr/bin/env python3.14
"""Longer-span cached M5 backtest for current trend, complement, and parallel sleeves.

Uses cached XAUUSD M5 CSV because the MT5 bridge rejects copy_rates_from_pos > 50,000 bars.
Aggregates H1 and M15 from M5 for HTF/M15 context.
"""
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from research_xauusd_trend_plus_complement_backtest import (
    OUT_DIR as BASE_OUT_DIR,
    Bar,
    Params,
    backtest,
    backtest_parallel_sleeves,
)

CSV_PATH = Path("/home/chain4655/Documents/backtest_reports/xauusd_5y_weekly_reset/data/XAUUSD_M5_5y_mt5.csv")
OUT_DIR = BASE_OUT_DIR / "long_cache"


def parse_time(value: str) -> dt.datetime:
    value = value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return dt.datetime.strptime(value, fmt)
        except ValueError:
            pass
    return dt.datetime.fromisoformat(value)


def load_m5_csv(path: Path) -> List[Bar]:
    bars: List[Bar] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bars.append(
                Bar(
                    time=parse_time(str(row["time"])),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    tick_volume=float(row.get("tick_volume") or 0.0),
                )
            )
    bars.sort(key=lambda bar: bar.time)
    return bars


def aggregate_bars(bars: List[Bar], minutes: int) -> List[Bar]:
    buckets: Dict[dt.datetime, List[Bar]] = {}
    for bar in bars:
        minute = (bar.time.minute // minutes) * minutes
        key = bar.time.replace(minute=minute, second=0, microsecond=0)
        if minutes == 60:
            key = bar.time.replace(minute=0, second=0, microsecond=0)
        buckets.setdefault(key, []).append(bar)
    out: List[Bar] = []
    for key in sorted(buckets):
        chunk = buckets[key]
        if not chunk:
            continue
        out.append(
            Bar(
                time=key,
                open=chunk[0].open,
                high=max(bar.high for bar in chunk),
                low=min(bar.low for bar in chunk),
                close=chunk[-1].close,
                tick_volume=sum(bar.tick_volume for bar in chunk),
            )
        )
    return out


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m5_bars = load_m5_csv(CSV_PATH)
    h1_bars = aggregate_bars(m5_bars, 60)
    m15_bars = aggregate_bars(m5_bars, 15)
    variants = [
        Params(name="current_trend_only_long", mode="trend_only"),
        Params(
            name="current_trend_only_chop_conservative_long",
            mode="trend_only",
            chop_gate="conservative_session",
        ),
        Params(name="complement_fb_sell_only_long", mode="complement_only"),
        Params(name="current_trend_plus_complement_long", mode="trend_plus_complement"),
        Params(
            name="current_trend_plus_complement_chop_conservative_long",
            mode="trend_plus_complement",
            chop_gate="conservative_session",
        ),
        Params(name="parallel_independent_sleeves_long", mode="parallel_sleeves"),
        Params(
            name="parallel_independent_sleeves_chop_conservative_long",
            mode="parallel_sleeves",
            chop_gate="conservative_session",
        ),
    ]
    summary_rows: List[dict] = []
    all_trades = []
    diagnostics = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source_csv": str(CSV_PATH),
        "m5_bars": len(m5_bars),
        "m5_start": m5_bars[0].time.isoformat(),
        "m5_end": m5_bars[-1].time.isoformat(),
        "h1_bars_aggregated": len(h1_bars),
        "m15_bars_aggregated": len(m15_bars),
        "note": "HTF/M15 context aggregated from cached M5; MT5 bridge rejected >50000 M5 bars.",
        "grouped": {},
    }
    for params in variants:
        if params.mode == "parallel_sleeves":
            trades, metrics = backtest_parallel_sleeves(m5_bars, h1_bars, m15_bars, params)
        else:
            trades, metrics = backtest(m5_bars, h1_bars, m15_bars, params)
        summary_rows.append({"strategy": params.name, **metrics})
        all_trades.extend(trades)
        diagnostics["grouped"][params.name] = {
            "signal_source_counts": dict(Counter(trade.signal_source for trade in trades)),
            "side_counts": dict(Counter(trade.side for trade in trades)),
            "month_counts": dict(Counter(trade.entry_time[:7] for trade in trades)),
        }
    fields = list(dict.fromkeys(key for row in summary_rows for key in row.keys()))
    write_csv(OUT_DIR / "trend_plus_complement_long_summary.csv", summary_rows, fields)
    write_csv(
        OUT_DIR / "trend_plus_complement_long_trades.csv",
        [dataclasses.asdict(trade) for trade in all_trades],
        [field.name for field in dataclasses.fields(all_trades[0])] if all_trades else [],
    )
    with (OUT_DIR / "trend_plus_complement_long_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)
    print(json.dumps({"summary": summary_rows, "diagnostics": diagnostics, "out_dir": str(OUT_DIR)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
