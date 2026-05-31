#!/usr/bin/env python3.14
"""Run expanded-population selectable-sample MT5 strategy optimization.

This wrapper refreshes a larger XAUUSD M15 SQLite universe, runs the existing DB-backed
optimizer for several sample windows, and emits a stability summary across windows.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path("/home/chain4655/Documents/Projects/MT5")
REPORT_ROOT = Path("/home/chain4655/Documents/backtest_reports/expanded_sample_strategy_selection")
OPT_ROOT = Path("/home/chain4655/Documents/backtest_reports/weekly_db_strategy_optimization")
OPT_SCRIPT = ROOT / "research_optimize_other_strategies_weekly_db.py"
SAMPLE_DAYS = [28, 60, 90, 180]
FETCH_DAYS = 365


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_optimizer(sample_days: int) -> Path:
    before = {p.name for p in OPT_ROOT.iterdir() if p.is_dir()} if OPT_ROOT.exists() else set()
    cmd = ["python3.14", str(OPT_SCRIPT), "--sample-days", str(sample_days), "--fetch-days", str(FETCH_DAYS)]
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1200)
    if proc.returncode != 0:
        raise RuntimeError(f"optimizer failed sample_days={sample_days}\n{proc.stdout[-4000:]}")
    try:
        payload = json.loads(proc.stdout[proc.stdout.index("{"):])
        out_dir = Path(payload["out_dir"])
    except Exception:
        after = sorted([p for p in OPT_ROOT.iterdir() if p.is_dir() and p.name not in before])
        if not after:
            raise RuntimeError(f"cannot locate optimizer output for sample_days={sample_days}\n{proc.stdout[-4000:]}")
        out_dir = after[-1]
    return out_dir


def fnum(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def inum(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def main() -> None:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPORT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs: Dict[int, str] = {}
    best_rows: List[Dict[str, Any]] = []
    top_rows: List[Dict[str, Any]] = []
    for sample_days in SAMPLE_DAYS:
        opt_dir = run_optimizer(sample_days)
        run_dirs[sample_days] = str(opt_dir)
        diag = json.loads((opt_dir / "diagnostics.json").read_text(encoding="utf-8"))
        for row in read_csv(opt_dir / "weekly_best_by_strategy.csv"):
            row = dict(row)
            row["sample_days"] = sample_days
            row["sample_start"] = diag["sample_start"]
            row["sample_end"] = diag["sample_end"]
            row["source_dir"] = str(opt_dir)
            best_rows.append(row)
        for row in read_csv(opt_dir / "weekly_param_grid_top50.csv"):
            row = dict(row)
            row["sample_days"] = sample_days
            row["source_dir"] = str(opt_dir)
            top_rows.append(row)

    # Stability by strategy family: reward positive windows, many trades, PF, and penalize DD.
    strategy_summary: List[Dict[str, Any]] = []
    for strategy in sorted({r["strategy"] for r in best_rows}):
        group = [r for r in best_rows if r["strategy"] == strategy]
        positive = [r for r in group if fnum(r["best_net_pnl"]) > 0]
        trades_total = sum(inum(r["best_trades"]) for r in group)
        avg_ret = sum(fnum(r["best_return_pct"]) for r in group) / max(len(group), 1)
        avg_pf = sum(min(fnum(r["best_pf"]), 5.0) for r in group) / max(len(group), 1)
        worst_dd = min(fnum(r["best_max_dd_pct"]) for r in group)
        stability_score = avg_ret + avg_pf * 0.5 + len(positive) * 1.0 - abs(worst_dd) * 0.65
        longest = max(group, key=lambda r: int(r["sample_days"]))
        best_rank = max(group, key=lambda r: fnum(r["best_rank_score"]))
        strategy_summary.append(
            {
                "strategy": strategy,
                "windows_tested": len(group),
                "positive_windows": len(positive),
                "total_best_trades": trades_total,
                "avg_return_pct": round(avg_ret, 3),
                "avg_capped_pf": round(avg_pf, 3),
                "worst_dd_pct": round(worst_dd, 3),
                "stability_score": round(stability_score, 5),
                "longest_sample_days": longest["sample_days"],
                "longest_best_param_id": longest["best_param_id"],
                "longest_best_net_pnl": longest["best_net_pnl"],
                "longest_best_return_pct": longest["best_return_pct"],
                "longest_best_pf": longest["best_pf"],
                "longest_best_dd_pct": longest["best_max_dd_pct"],
                "longest_best_params_json": longest["best_params_json"],
                "best_rank_window_days": best_rank["sample_days"],
                "best_rank_param_id": best_rank["best_param_id"],
            }
        )
    strategy_summary.sort(key=lambda r: (float(r["stability_score"]), int(r["positive_windows"])), reverse=True)

    # Parameter stability: only compare top50 rows. Same param_id recurring across windows is more robust.
    param_summary: List[Dict[str, Any]] = []
    for key in sorted({(r["strategy"], r["param_id"]) for r in top_rows}):
        strategy, param_id = key
        group = [r for r in top_rows if r["strategy"] == strategy and r["param_id"] == param_id]
        if len(group) < 2:
            continue
        positive = [r for r in group if fnum(r["net_pnl"]) > 0]
        avg_ret = sum(fnum(r["return_pct"]) for r in group) / len(group)
        worst_dd = min(fnum(r["max_dd_pct"]) for r in group)
        longest = max(group, key=lambda r: int(r["sample_days"]))
        param_summary.append(
            {
                "strategy": strategy,
                "param_id": param_id,
                "top50_windows": len(group),
                "positive_top50_windows": len(positive),
                "avg_return_pct": round(avg_ret, 3),
                "worst_dd_pct": round(worst_dd, 3),
                "longest_sample_days": longest["sample_days"],
                "longest_net_pnl": longest["net_pnl"],
                "longest_return_pct": longest["return_pct"],
                "longest_pf": longest["profit_factor"],
                "params_json": longest["params_json"],
            }
        )
    param_summary.sort(
        key=lambda r: (int(r["positive_top50_windows"]), int(r["top50_windows"]), fnum(r["avg_return_pct"]), -abs(fnum(r["worst_dd_pct"]))),
        reverse=True,
    )

    write_csv(out_dir / "best_by_strategy_all_samples.csv", best_rows)
    write_csv(out_dir / "strategy_stability_summary.csv", strategy_summary)
    write_csv(out_dir / "recurring_top_params.csv", param_summary)
    diagnostics = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "sample_days": SAMPLE_DAYS,
        "fetch_days": FETCH_DAYS,
        "optimizer_runs": run_dirs,
        "note": "Expanded population uses MT5 XAUUSD M15 history up to 365 days where available. Strategy rankings are still in-sample; use rolling out-of-sample before live switch.",
    }
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Expanded Sample Strategy Selection",
        "",
        f"Generated: {diagnostics['generated_at']}",
        f"Sample windows: {SAMPLE_DAYS}",
        f"Fetch days requested: {FETCH_DAYS}",
        "",
        "## Strategy stability ranking",
    ]
    for row in strategy_summary:
        lines.append(
            f"- {row['strategy']}: score={row['stability_score']} positive={row['positive_windows']}/{row['windows_tested']} "
            f"avg_ret={row['avg_return_pct']}% worst_DD={row['worst_dd_pct']}% longest={row['longest_sample_days']}d "
            f"param={row['longest_best_param_id']} ret={row['longest_best_return_pct']}% PF={row['longest_best_pf']}"
        )
    lines.extend([
        "",
        "## Recurring top params",
    ])
    for row in param_summary[:20]:
        lines.append(
            f"- {row['strategy']} {row['param_id']}: top50_windows={row['top50_windows']} positive={row['positive_top50_windows']} "
            f"avg_ret={row['avg_return_pct']}% longest={row['longest_sample_days']}d ret={row['longest_return_pct']}%"
        )
        lines.append(f"  - params: `{row['params_json']}`")
    lines.extend([
        "",
        "## Files",
        "- `strategy_stability_summary.csv`",
        "- `recurring_top_params.csv`",
        "- `best_by_strategy_all_samples.csv`",
        "- `diagnostics.json`",
    ])
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "strategy_summary": strategy_summary[:10], "param_summary": param_summary[:10], "run_dirs": run_dirs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
