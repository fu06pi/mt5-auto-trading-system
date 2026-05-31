#!/usr/bin/env python3
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

INITIAL_BALANCE = 10000.0
TARGET_PCT = 0.05
MAX_DAILY_LOSS_PCT = 0.03
MAX_OVERALL_LOSS_PCT = 0.05
TARGET_BALANCE = INITIAL_BALANCE * (1.0 + TARGET_PCT)
DAILY_FLOOR_DELTA = INITIAL_BALANCE * MAX_DAILY_LOSS_PCT
OVERALL_FLOOR = INITIAL_BALANCE * (1.0 - MAX_OVERALL_LOSS_PCT)
RISK_GRID = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05]
OUT_ROOT = Path("/home/chain4655/Documents/backtest_reports/profirm_challenge_sim")


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    label: str
    path: Path
    strategy_col: str
    period_col: Optional[str] = None
    include_filter: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class SimResult:
    dataset: str
    strategy: str
    period: str
    risk_pct: float
    status: str
    trades_used: int
    total_trades_available: int
    start_time: str
    end_time: str
    final_balance: float
    net_return_pct: float
    max_dd_pct: float
    max_daily_loss_pct: float
    min_balance: float
    longest_loss_streak: int
    pass_time: str
    fail_time: str
    fail_reason: str
    profit_factor: float
    expectancy_r: float
    source_path: str


def dataset_specs() -> List[DatasetSpec]:
    return [
        DatasetSpec(
            "london_asian_v2",
            Path("/home/chain4655/Documents/backtest_reports/london_asian_breakout_v2/20260520_230637/all_trades.csv"),
            "config",
            "period",
        ),
        DatasetSpec(
            "twitter_xauusd_candidates",
            Path("/home/chain4655/Documents/backtest_reports/twitter_xauusd_candidates/20260520_225323/all_trades.csv"),
            "strategy",
            "period",
        ),
        DatasetSpec(
            "mt5_strategy_library",
            Path("/home/chain4655/Documents/Projects/MT5/backtest_reports/doomsday_trades.csv"),
            "strategy",
        ),
        DatasetSpec(
            "mt5_strategy_library",
            Path("/home/chain4655/Documents/Projects/MT5/backtest_reports/bollinger_edge_squeeze_trades.csv"),
            "strategy",
        ),
        DatasetSpec(
            "mt5_strategy_library",
            Path("/home/chain4655/Documents/Projects/MT5/backtest_reports/trend_following_trades.csv"),
            "strategy",
        ),
        DatasetSpec(
            "mt5_strategy_library",
            Path("/home/chain4655/Documents/Projects/MT5/backtest_reports/asia_london_breakout_ep1_trades.csv"),
            "strategy",
        ),
        DatasetSpec(
            "mt5_strategy_library",
            Path("/home/chain4655/Documents/Projects/MT5/backtest_reports/momentum_surfer_trades.csv"),
            "strategy",
        ),
        DatasetSpec(
            "mt5_strategy_library",
            Path("/home/chain4655/Documents/Projects/MT5/backtest_reports/meta_regime_switch_trades.csv"),
            "strategy",
        ),
        DatasetSpec(
            "po3_smt_proxy",
            Path("/home/chain4655/Documents/backtest_reports/po3_smt_proxy/all_trades.csv"),
            "strategy",
        ),
        DatasetSpec(
            "liquidity_trap_proxy",
            Path("/home/chain4655/Documents/backtest_reports/liquidity_trap_proxy/all_trades.csv"),
            "strategy",
        ),
        DatasetSpec(
            "trident_pattern_proxy",
            Path("/home/chain4655/Documents/backtest_reports/trident_pattern_proxy/all_trades.csv"),
            "strategy",
        ),
        DatasetSpec(
            "larry_outside_bar_xauusd",
            Path("/home/chain4655/Documents/backtest_reports/larry_outside_bar_xauusd/trades.csv"),
            "strategy",
        ),
    ]


def normalize_dataframe(spec: DatasetSpec) -> pd.DataFrame:
    if not spec.path.exists():
        return pd.DataFrame()
    df = pd.read_csv(spec.path)
    if df.empty or "entry_time" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    if "exit_time" in df.columns:
        df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
    else:
        df["exit_time"] = df["entry_time"]
    df = df.dropna(subset=["entry_time", "exit_time"])
    if spec.strategy_col not in df.columns:
        df["strategy"] = spec.path.stem
        strategy_col = "strategy"
    else:
        strategy_col = spec.strategy_col
    df["strategy_name"] = df[strategy_col].astype(str)
    if spec.period_col and spec.period_col in df.columns:
        df["period_name"] = df[spec.period_col].astype(str)
    else:
        df["period_name"] = "source_full"
    if "r_multiple" not in df.columns:
        return pd.DataFrame()
    df["r_multiple"] = pd.to_numeric(df["r_multiple"], errors="coerce")
    df = df.dropna(subset=["r_multiple"])
    df = df.sort_values(["entry_time", "exit_time"]).reset_index(drop=True)
    df["source_path"] = str(spec.path)
    df["dataset"] = spec.label
    return df


def simulate(trades: pd.DataFrame, risk_pct: float, dataset: str, strategy: str, period: str, source_path: str) -> SimResult:
    balance = INITIAL_BALANCE
    peak = balance
    min_balance = balance
    max_dd = 0.0
    max_daily_loss = 0.0
    current_day: Optional[dt.date] = None
    day_start_balance = balance
    loss_streak = 0
    longest_loss_streak = 0
    gross_profit = 0.0
    gross_loss = 0.0
    used = 0
    pass_time = ""
    fail_time = ""
    fail_reason = ""
    status = "NO_PASS"
    end_time = ""

    ordered = trades.sort_values(["exit_time", "entry_time"]).reset_index(drop=True)
    start_time = ordered.iloc[0].entry_time.isoformat() if len(ordered) else ""
    for _, row in ordered.iterrows():
        exit_time = row.exit_time.to_pydatetime()
        trade_day = exit_time.date()
        if current_day != trade_day:
            current_day = trade_day
            day_start_balance = balance
        pnl = balance * risk_pct * float(row.r_multiple)
        balance += pnl
        used += 1
        end_time = exit_time.isoformat()
        if pnl > 0:
            gross_profit += pnl
            loss_streak = 0
        else:
            gross_loss += -pnl
            loss_streak += 1
            longest_loss_streak = max(longest_loss_streak, loss_streak)
        peak = max(peak, balance)
        min_balance = min(min_balance, balance)
        max_dd = min(max_dd, balance / peak - 1.0)
        daily_loss = max(0.0, day_start_balance - balance)
        max_daily_loss = max(max_daily_loss, daily_loss / INITIAL_BALANCE)
        if balance <= OVERALL_FLOOR:
            status = "FAIL"
            fail_time = end_time
            fail_reason = "max_overall_loss"
            break
        if daily_loss >= DAILY_FLOOR_DELTA:
            status = "FAIL"
            fail_time = end_time
            fail_reason = "max_daily_loss"
            break
        if balance >= TARGET_BALANCE:
            status = "PASS"
            pass_time = end_time
            break
    if used == 0:
        status = "NO_TRADES"
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    return SimResult(
        dataset=dataset,
        strategy=strategy,
        period=period,
        risk_pct=round(risk_pct * 100, 3),
        status=status,
        trades_used=used,
        total_trades_available=len(ordered),
        start_time=start_time,
        end_time=end_time,
        final_balance=round(balance, 2),
        net_return_pct=round((balance / INITIAL_BALANCE - 1.0) * 100, 3),
        max_dd_pct=round(max_dd * 100, 3),
        max_daily_loss_pct=round(max_daily_loss * 100, 3),
        min_balance=round(min_balance, 2),
        longest_loss_streak=longest_loss_streak,
        pass_time=pass_time,
        fail_time=fail_time,
        fail_reason=fail_reason,
        profit_factor=round(profit_factor, 3),
        expectancy_r=round(float(ordered["r_multiple"].mean()) if len(ordered) else 0.0, 3),
        source_path=source_path,
    )


def summarize_strategy(df: pd.DataFrame) -> Dict[str, object]:
    wins = df[df.r_multiple > 0]
    losses = df[df.r_multiple <= 0]
    gp = float(wins.r_multiple.sum())
    gl = -float(losses.r_multiple.sum())
    return {
        "trades": len(df),
        "start": df.entry_time.min().isoformat() if len(df) else "",
        "end": df.exit_time.max().isoformat() if len(df) else "",
        "win_rate_pct": round(len(wins) / len(df) * 100, 2) if len(df) else 0.0,
        "profit_factor_r": round(gp / gl, 3) if gl > 0 else (999.0 if gp > 0 else 0.0),
        "expectancy_r": round(float(df.r_multiple.mean()) if len(df) else 0.0, 3),
        "avg_win_r": round(float(wins.r_multiple.mean()) if len(wins) else 0.0, 3),
        "avg_loss_r": round(float(losses.r_multiple.mean()) if len(losses) else 0.0, 3),
    }


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def result_sort_key(row: Dict[str, object]) -> tuple:
    status_rank = {"PASS": 3, "NO_PASS": 2, "FAIL": 1, "NO_TRADES": 0}.get(str(row["status"]), 0)
    return (
        status_rank,
        float(row["net_return_pct"]),
        -abs(float(row["max_daily_loss_pct"])),
        -abs(float(row["max_dd_pct"])),
        float(row["profit_factor"]),
    )


def build_report(out_dir: Path, results: List[Dict[str, object]], best_rows: List[Dict[str, object]], passes: List[Dict[str, object]]) -> str:
    lines = [
        "# ProFirm Challenge Simulation",
        "",
        f"Output: `{out_dir}`",
        "",
        "## Rules",
        f"- Initial balance: {INITIAL_BALANCE:.0f}",
        f"- Target: {TARGET_PCT * 100:.1f}% = {TARGET_BALANCE:.0f}",
        f"- Max daily loss: {MAX_DAILY_LOSS_PCT * 100:.1f}% = {DAILY_FLOOR_DELTA:.0f}",
        f"- Max overall loss: {MAX_OVERALL_LOSS_PCT * 100:.1f}% = floor {OVERALL_FLOOR:.0f}",
        "- Simulation uses each trade's `r_multiple`; PnL = current_balance * risk_pct * r_multiple.",
        "- Closed-trade daily loss only; no intratrade floating drawdown because most backtest logs do not export tick-level adverse excursion.",
        "",
        "## Pass candidates",
    ]
    if passes:
        for row in passes[:30]:
            lines.append(
                f"- {row['dataset']} / {row['strategy']} / {row['period']} @ {row['risk_pct']}% risk: "
                f"PASS in {row['trades_used']}/{row['total_trades_available']} trades, "
                f"return={row['net_return_pct']}%, maxDaily={row['max_daily_loss_pct']}%, "
                f"DD={row['max_dd_pct']}%, PF={row['profit_factor']}"
            )
    else:
        lines.append("- None passed under the tested risk grid.")
    lines += ["", "## Best row per strategy/period"]
    for row in best_rows[:80]:
        lines.append(
            f"- {row['dataset']} / {row['strategy']} / {row['period']}: best @ {row['risk_pct']}% risk, "
            f"status={row['status']}, return={row['net_return_pct']}%, maxDaily={row['max_daily_loss_pct']}%, "
            f"DD={row['max_dd_pct']}%, trades={row['trades_used']}/{row['total_trades_available']}, PF={row['profit_factor']}"
        )
    lines += [
        "",
        "## Files",
        "- `all_prop_sim_results.csv`: all strategy x risk simulations.",
        "- `passes.csv`: only rows that hit +5% before violating drawdown rules.",
        "- `best_by_strategy.csv`: best risk row per strategy/period.",
        "- `strategy_r_summary.csv`: raw R-multiple stats before challenge sizing.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    all_frames = [normalize_dataframe(spec) for spec in dataset_specs()]
    all_frames = [frame for frame in all_frames if not frame.empty]
    if not all_frames:
        raise RuntimeError("No usable trade logs found")
    all_data = pd.concat(all_frames, ignore_index=True)
    results: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []
    group_cols = ["dataset", "strategy_name", "period_name", "source_path"]
    for (dataset, strategy, period, source_path), group in all_data.groupby(group_cols, sort=True):
        group = group.sort_values(["entry_time", "exit_time"]).reset_index(drop=True)
        summary = summarize_strategy(group)
        summaries.append(
            {
                "dataset": dataset,
                "strategy": strategy,
                "period": period,
                "source_path": source_path,
                **summary,
            }
        )
        for risk in RISK_GRID:
            result = simulate(group, risk, dataset, strategy, period, source_path)
            results.append(dataclasses.asdict(result))
    passes = sorted([row for row in results if row["status"] == "PASS"], key=result_sort_key, reverse=True)
    best_rows: List[Dict[str, object]] = []
    result_df = pd.DataFrame(results)
    for (_dataset, _strategy, _period), group in result_df.groupby(["dataset", "strategy", "period"], sort=True):
        group_rows = group.to_dict("records")
        best_rows.append(sorted(group_rows, key=result_sort_key, reverse=True)[0])
    best_rows = sorted(best_rows, key=result_sort_key, reverse=True)
    write_csv(out_dir / "all_prop_sim_results.csv", results)
    write_csv(out_dir / "passes.csv", passes)
    write_csv(out_dir / "best_by_strategy.csv", best_rows)
    write_csv(out_dir / "strategy_r_summary.csv", summaries)
    (out_dir / "REPORT.md").write_text(build_report(out_dir, results, best_rows, passes), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "rules": {
                    "initial_balance": INITIAL_BALANCE,
                    "target_pct": TARGET_PCT,
                    "max_daily_loss_pct": MAX_DAILY_LOSS_PCT,
                    "max_overall_loss_pct": MAX_OVERALL_LOSS_PCT,
                    "risk_grid": RISK_GRID,
                },
                "pass_count": len(passes),
                "top_passes": passes[:30],
                "top_best_rows": best_rows[:30],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(out_dir), "pass_count": len(passes), "top_passes": passes[:10], "top_best_rows": best_rows[:10]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
