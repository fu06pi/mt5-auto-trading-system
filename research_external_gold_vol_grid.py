#!/usr/bin/env python3
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import yfinance as yf

from research_low_vol_xauusd_active_grid import Bar, Trade, backtest_strategy, strategy_grid

OUT_ROOT = Path("/home/chain4655/Documents/backtest_reports/xauusd_external_vol_grid")
LOCAL_XAU_M5 = Path("/home/chain4655/Documents/backtest_reports/twitter_xauusd_candidates/20260520_225323/input_xauusd_m5.csv")


def week_key(ts: dt.datetime) -> str:
    year, week, _ = ts.isocalendar()
    return f"{year}-W{week:02d}"


def flatten_yf(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(-1, axis=1)
    data = data.reset_index()
    data.columns = [str(c).strip().lower().replace(" ", "_") for c in data.columns]
    time_col = "datetime" if "datetime" in data.columns else "date"
    data = data.rename(columns={time_col: "time"})
    return data


def fetch_yahoo_bars(symbol: str = "GC=F", period: str = "60d", interval: str = "5m") -> Tuple[List[Bar], Path]:
    data = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)
    if data.empty:
        raise RuntimeError(f"Yahoo returned no data for {symbol} {period} {interval}")
    data = flatten_yf(data)
    data["time"] = pd.to_datetime(data["time"], utc=True).dt.tz_convert(None)
    data = data.sort_values("time")
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_dir = OUT_ROOT / stamp
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{symbol.replace('=', '').replace('^', '')}_{interval}_{period}.csv"
    data.to_csv(cache_path, index=False)
    bars: List[Bar] = []
    for row in data.itertuples(index=False):
        t = getattr(row, "time").to_pydatetime()
        bars.append(
            Bar(
                time=t,
                open=float(getattr(row, "open")),
                high=float(getattr(row, "high")),
                low=float(getattr(row, "low")),
                close=float(getattr(row, "close")),
                volume=float(getattr(row, "volume", 0.0) or 0.0),
                week=week_key(t),
            )
        )
    return bars, cache_dir


def weekly_diagnostics(bars: Sequence[Bar]) -> Tuple[List[Dict[str, object]], set[str], set[str]]:
    grouped: Dict[str, List[Bar]] = {}
    for b in bars:
        grouped.setdefault(b.week, []).append(b)
    rows: List[Dict[str, object]] = []
    range_weeks: set[str] = set()
    realized_weeks: set[str] = set()
    # 5m bars per trading year proxy: 288/day * 252. Futures sessions are not exact; this is a regime classifier only.
    ann_factor = math.sqrt(288.0 * 252.0)
    for wk, g in sorted(grouped.items()):
        if len(g) < 500:
            sample_ok = False
        else:
            sample_ok = True
        open_px = g[0].open
        high = max(b.high for b in g)
        low = min(b.low for b in g)
        close = g[-1].close
        range_pct = (high - low) / max(open_px, 1e-9) * 100.0
        rets = [math.log(g[i].close / g[i - 1].close) for i in range(1, len(g)) if g[i - 1].close > 0 and g[i].close > 0]
        if len(rets) >= 30:
            mean = sum(rets) / len(rets)
            var = sum((x - mean) ** 2 for x in rets) / (len(rets) - 1)
            realized_ann_pct = math.sqrt(var) * ann_factor * 100.0
        else:
            realized_ann_pct = 0.0
        in_range_bucket = sample_ok and 20.0 <= range_pct < 50.0
        in_realized_bucket = sample_ok and 20.0 <= realized_ann_pct < 50.0
        if in_range_bucket:
            range_weeks.add(wk)
        if in_realized_bucket:
            realized_weeks.add(wk)
        rows.append(
            {
                "week": wk,
                "start": g[0].time.isoformat(sep=" "),
                "end": g[-1].time.isoformat(sep=" "),
                "bars": len(g),
                "open": round(open_px, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "range_pct": round(range_pct, 3),
                "realized_vol_ann_pct": round(realized_ann_pct, 3),
                "range_20_50": in_range_bucket,
                "realized_ann_20_50": in_realized_bucket,
                "sample_ok": sample_ok,
            }
        )
    return rows, range_weeks, realized_weeks


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_grid(bars: Sequence[Bar], weeks: set[str], out_dir: Path, label: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    selected_trades: List[Dict[str, object]] = []
    if not weeks:
        write_csv(out_dir / f"{label}_grid_summary.csv", rows)
        write_csv(out_dir / f"{label}_top10_trades.csv", selected_trades)
        return rows
    for strat in strategy_grid():
        trades, metrics = backtest_strategy(bars, weeks, strat)
        metrics = dict(metrics)
        metrics["bucket"] = label
        rows.append(metrics)
    rows = sorted(rows, key=lambda r: (float(r["return_pct"]), float(r["profit_factor"]), int(r["trades"])), reverse=True)
    write_csv(out_dir / f"{label}_grid_summary.csv", rows)
    top_ids = {str(r["param_id"]) for r in rows[:10]}
    for strat in strategy_grid():
        if strat.param_id not in top_ids:
            continue
        trades, _ = backtest_strategy(bars, weeks, strat)
        for t in trades:
            row = dataclasses.asdict(t)
            row["entry_time"] = t.entry_time.isoformat(sep=" ")
            row["exit_time"] = t.exit_time.isoformat(sep=" ")
            row["gross_pnl"] = round(t.gross_pnl, 2)
            row["commission"] = round(t.commission, 2)
            row["net_pnl"] = round(t.net_pnl, 2)
            row["r_multiple"] = round(t.r_multiple, 4)
            selected_trades.append(row)
    write_csv(out_dir / f"{label}_top10_trades.csv", selected_trades)
    return rows


def compare_external_to_local(external: Sequence[Bar]) -> Dict[str, object]:
    if not LOCAL_XAU_M5.exists():
        return {"status": "missing_local_reference"}
    local = pd.read_csv(LOCAL_XAU_M5)
    local["time"] = pd.to_datetime(local["time"])
    local = local.sort_values("time")[["time", "close"]].rename(columns={"close": "local_close"})
    ext_base = pd.DataFrame([{"time": b.time, "ext_close": b.close} for b in external]).sort_values("time")
    candidates: List[Dict[str, object]] = []
    for offset_hours in range(-12, 13):
        ext = ext_base.copy()
        ext["time"] = ext["time"] + pd.Timedelta(hours=offset_hours)
        merged = pd.merge_asof(
            local,
            ext,
            on="time",
            tolerance=pd.Timedelta("2min"),
            direction="nearest",
        ).dropna()
        if len(merged) < 100:
            continue
        merged["local_ret"] = merged["local_close"].pct_change()
        merged["ext_ret"] = merged["ext_close"].pct_change()
        ret_corr = float(merged[["local_ret", "ext_ret"]].dropna().corr().iloc[0, 1])
        basis = merged["local_close"] - merged["ext_close"]
        candidates.append(
            {
                "offset_hours": offset_hours,
                "overlap_rows": int(len(merged)),
                "overlap_start": merged["time"].min().isoformat(),
                "overlap_end": merged["time"].max().isoformat(),
                "return_corr": round(ret_corr, 4),
                "mean_price_basis_local_minus_gc": round(float(basis.mean()), 3),
                "median_abs_basis": round(float(basis.abs().median()), 3),
                "max_abs_basis": round(float(basis.abs().max()), 3),
            }
        )
    if not candidates:
        return {"status": "insufficient_overlap", "overlap_rows": 0}
    best = sorted(candidates, key=lambda x: float(x["return_corr"]), reverse=True)[0]
    best["status"] = "ok"
    best["tested_offsets_hours"] = [-12, 12]
    return best


def main() -> None:
    bars, out_dir = fetch_yahoo_bars()
    weekly_rows, range_weeks, realized_weeks = weekly_diagnostics(bars)
    write_csv(out_dir / "weekly_volatility_external_gc.csv", weekly_rows)
    range_summary = run_grid(bars, range_weeks, out_dir, "range_20_50")
    realized_summary = run_grid(bars, realized_weeks, out_dir, "realized_ann_20_50")
    comparison = compare_external_to_local(bars)
    top_range = range_summary[:10]
    top_realized = realized_summary[:10]
    report = [
        "# External Gold Data Volatility Bucket Backtest",
        "",
        "Source: Yahoo Finance `GC=F` COMEX gold futures 5m, not MT5 bridge/history.",
        f"Bars: {len(bars)} | range: {bars[0].time} → {bars[-1].time}",
        "TradingView CDP check: unavailable on localhost:9222 in this session; used Yahoo Finance as independent external source.",
        f"Weekly high-low/open 20-50% weeks: {len(range_weeks)}",
        f"Annualized 5m realized-vol 20-50% weeks: {len(realized_weeks)}",
        "",
        "## Data realism checks",
        f"- External vs local M5 overlap: `{json.dumps(comparison, ensure_ascii=False)}`",
        "- GC futures is not broker XAUUSD spot: execution/spread/swap differ; signal timing is useful, exact PnL is proxy only.",
        "- 5m Yahoo bars are delayed/aggregated futures bars, not tick-true fills.",
        "- Backtest still uses single-bar OHLC path assumption: if SL and TP hit within same bar, SL is conservatively checked first.",
        "",
        "## Top range_20_50 results",
    ]
    if not top_range:
        report.append("- No weekly high-low/open 20-50% weeks found, so no valid backtest for this exact bucket.")
    else:
        for i, row in enumerate(top_range, 1):
            report.append(f"{i}. {row['strategy']}: return={row['return_pct']}%, PF={row['profit_factor']}, trades={row['trades']}, DD={row['max_dd_pct']}% | {row['param_id']}")
    report.append("")
    report.append("## Top realized_ann_20_50 results")
    if not top_realized:
        report.append("- No annualized realized-vol 20-50% weeks found.")
    else:
        for i, row in enumerate(top_realized, 1):
            report.append(f"{i}. {row['strategy']}: return={row['return_pct']}%, PF={row['profit_factor']}, trades={row['trades']}, DD={row['max_dd_pct']}% | {row['param_id']}")
    report.extend([
        "",
        "## Files",
        "- `weekly_volatility_external_gc.csv`",
        "- `range_20_50_grid_summary.csv` / `range_20_50_top10_trades.csv`",
        "- `realized_ann_20_50_grid_summary.csv` / `realized_ann_20_50_top10_trades.csv`",
    ])
    (out_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    payload = {
        "out_dir": str(out_dir),
        "bars": len(bars),
        "start": bars[0].time.isoformat(),
        "end": bars[-1].time.isoformat(),
        "range_20_50_week_count": len(range_weeks),
        "realized_ann_20_50_week_count": len(realized_weeks),
        "external_vs_local": comparison,
        "top_range": top_range,
        "top_realized": top_realized,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
