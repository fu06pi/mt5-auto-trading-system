#!/usr/bin/env python3
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

import profirm_challenge_simulator as base

OUT_ROOT = Path("/home/chain4655/Documents/backtest_reports/profirm_portfolio_challenge_sim")
INITIAL_BALANCE = 10000.0
TARGET_BALANCE = 10500.0
OVERALL_FLOOR = 9500.0
DAILY_FAIL_DELTA = 300.0


@dataclasses.dataclass(frozen=True)
class Leg:
    dataset: str
    strategy: str
    period: str
    risk_pct: float
    source_contains: str = ""


@dataclasses.dataclass(frozen=True)
class PortfolioSpec:
    name: str
    legs: Tuple[Leg, ...]
    daily_soft_stop_pct: float = 0.02
    profit_lock_start_pct: float = 0.04
    profit_lock_multiplier: float = 0.5
    max_trades_per_day: int = 6


def load_all_trades() -> pd.DataFrame:
    frames = [base.normalize_dataframe(spec) for spec in base.dataset_specs()]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise RuntimeError("No trade logs found")
    data = pd.concat(frames, ignore_index=True)
    data["portfolio_key"] = (
        data["dataset"].astype(str)
        + "|"
        + data["strategy_name"].astype(str)
        + "|"
        + data["period_name"].astype(str)
    )
    return data


def select_leg(data: pd.DataFrame, leg: Leg) -> pd.DataFrame:
    mask = (
        data["dataset"].eq(leg.dataset)
        & data["strategy_name"].eq(leg.strategy)
        & data["period_name"].eq(leg.period)
    )
    if leg.source_contains:
        mask = mask & data["source_path"].str.contains(leg.source_contains, regex=False)
    out = data[mask].copy()
    out["leg_risk_pct"] = leg.risk_pct
    out["leg_name"] = f"{leg.dataset}:{leg.strategy}:{leg.period}"
    return out


def build_portfolios() -> List[PortfolioSpec]:
    return [
        PortfolioSpec(
            "library_core_trend_doomsday_surfer_balanced",
            (
                Leg("mt5_strategy_library", "meta::trend_following", "source_full", 0.0075),
                Leg("mt5_strategy_library", "doomsday", "source_full", 0.0050),
                Leg("mt5_strategy_library", "momentum_surfer", "source_full", 0.0050),
            ),
        ),
        PortfolioSpec(
            "library_core_trend_doomsday_surfer_equal_075",
            (
                Leg("mt5_strategy_library", "meta::trend_following", "source_full", 0.0075),
                Leg("mt5_strategy_library", "doomsday", "source_full", 0.0075),
                Leg("mt5_strategy_library", "momentum_surfer", "source_full", 0.0075),
            ),
        ),
        PortfolioSpec(
            "library_core_trend_surfer_only",
            (
                Leg("mt5_strategy_library", "meta::trend_following", "source_full", 0.0100),
                Leg("mt5_strategy_library", "momentum_surfer", "source_full", 0.0050),
            ),
        ),
        PortfolioSpec(
            "library_core_doomsday_surfer_only",
            (
                Leg("mt5_strategy_library", "doomsday", "source_full", 0.0075),
                Leg("mt5_strategy_library", "momentum_surfer", "source_full", 0.0050),
            ),
        ),
        PortfolioSpec(
            "momentum_surfer_favorite_safe",
            (Leg("mt5_strategy_library", "momentum_surfer", "source_full", 0.0075),),
        ),
        PortfolioSpec(
            "momentum_surfer_favorite_standard",
            (Leg("mt5_strategy_library", "momentum_surfer", "source_full", 0.0100),),
        ),
        PortfolioSpec(
            "po3_proxy_plus_surfer_safe",
            (
                Leg("po3_smt_proxy", "po3_1400_1700_no_smt", "source_full", 0.0050),
                Leg("mt5_strategy_library", "momentum_surfer", "source_full", 0.0050),
            ),
        ),
        PortfolioSpec(
            "po3_proxy_plus_surfer_balanced",
            (
                Leg("po3_smt_proxy", "po3_1330_1600_no_smt", "source_full", 0.0075),
                Leg("mt5_strategy_library", "momentum_surfer", "source_full", 0.0050),
            ),
        ),
        PortfolioSpec(
            "po3_proxy_trend_surfer",
            (
                Leg("po3_smt_proxy", "po3_1400_1700_no_smt", "source_full", 0.0050),
                Leg("mt5_strategy_library", "meta::trend_following", "source_full", 0.0075),
                Leg("mt5_strategy_library", "momentum_surfer", "source_full", 0.0050),
            ),
        ),
        PortfolioSpec(
            "aggressive_trend_doomsday_surfer_1pct",
            (
                Leg("mt5_strategy_library", "meta::trend_following", "source_full", 0.0100),
                Leg("mt5_strategy_library", "doomsday", "source_full", 0.0100),
                Leg("mt5_strategy_library", "momentum_surfer", "source_full", 0.0100),
            ),
            daily_soft_stop_pct=0.018,
            profit_lock_start_pct=0.035,
            profit_lock_multiplier=0.4,
            max_trades_per_day=4,
        ),
    ]


def combine_portfolio(data: pd.DataFrame, spec: PortfolioSpec) -> pd.DataFrame:
    frames = [select_leg(data, leg) for leg in spec.legs]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    combo = pd.concat(frames, ignore_index=True)
    combo = combo.sort_values(["entry_time", "exit_time", "leg_name"]).reset_index(drop=True)
    # Avoid exact duplicate rows from overlapping sources in the same portfolio.
    combo = combo.drop_duplicates(
        subset=["entry_time", "exit_time", "side", "entry", "exit", "r_multiple", "leg_name"]
    ).reset_index(drop=True)
    return combo


def simulate_sequence(trades: pd.DataFrame, spec: PortfolioSpec, start_idx: int) -> Dict[str, object]:
    ordered = trades.iloc[start_idx:].sort_values(["entry_time", "exit_time", "leg_name"]).reset_index(drop=True)
    balance = INITIAL_BALANCE
    peak = balance
    min_balance = balance
    max_dd = 0.0
    max_daily_loss = 0.0
    current_day: Optional[dt.date] = None
    day_start_balance = balance
    day_loss = 0.0
    day_trades = 0
    used = 0
    skipped_soft_stop = 0
    skipped_trade_cap = 0
    leg_pnl: Dict[str, float] = {}
    leg_trades: Dict[str, int] = {}
    status = "NO_PASS"
    fail_reason = ""
    end_time = ""
    pass_time = ""
    fail_time = ""
    start_time = ordered.iloc[0].entry_time.isoformat() if len(ordered) else ""

    for row in ordered.itertuples(index=False):
        entry_time = row.entry_time.to_pydatetime()
        day = entry_time.date()
        if current_day != day:
            current_day = day
            day_start_balance = balance
            day_loss = 0.0
            day_trades = 0
        if day_loss >= INITIAL_BALANCE * spec.daily_soft_stop_pct:
            skipped_soft_stop += 1
            continue
        if day_trades >= spec.max_trades_per_day:
            skipped_trade_cap += 1
            continue
        risk_pct = float(row.leg_risk_pct)
        if balance >= INITIAL_BALANCE * (1.0 + spec.profit_lock_start_pct):
            risk_pct *= spec.profit_lock_multiplier
        pnl = balance * risk_pct * float(row.r_multiple)
        balance += pnl
        used += 1
        day_trades += 1
        end_time = row.exit_time.isoformat()
        leg = str(row.leg_name)
        leg_pnl[leg] = leg_pnl.get(leg, 0.0) + pnl
        leg_trades[leg] = leg_trades.get(leg, 0) + 1
        peak = max(peak, balance)
        min_balance = min(min_balance, balance)
        max_dd = min(max_dd, balance / peak - 1.0)
        day_loss = max(0.0, day_start_balance - balance)
        max_daily_loss = max(max_daily_loss, day_loss / INITIAL_BALANCE)
        if balance <= OVERALL_FLOOR:
            status = "FAIL"
            fail_reason = "max_overall_loss"
            fail_time = end_time
            break
        if day_loss >= DAILY_FAIL_DELTA:
            status = "FAIL"
            fail_reason = "max_daily_loss"
            fail_time = end_time
            break
        if balance >= TARGET_BALANCE:
            status = "PASS"
            pass_time = end_time
            break

    return {
        "portfolio": spec.name,
        "start_idx": start_idx,
        "status": status,
        "trades_used": used,
        "trades_available": len(ordered),
        "start_time": start_time,
        "end_time": end_time,
        "pass_time": pass_time,
        "fail_time": fail_time,
        "fail_reason": fail_reason,
        "final_balance": round(balance, 2),
        "net_return_pct": round((balance / INITIAL_BALANCE - 1.0) * 100.0, 3),
        "max_dd_pct": round(max_dd * 100.0, 3),
        "max_daily_loss_pct": round(max_daily_loss * 100.0, 3),
        "min_balance": round(min_balance, 2),
        "skipped_soft_stop": skipped_soft_stop,
        "skipped_trade_cap": skipped_trade_cap,
        "leg_pnl_json": json.dumps({k: round(v, 2) for k, v in leg_pnl.items()}, ensure_ascii=False),
        "leg_trades_json": json.dumps(leg_trades, ensure_ascii=False),
    }


def summarize(rows: List[Dict[str, object]], spec: PortfolioSpec, total_trades: int) -> Dict[str, object]:
    df = pd.DataFrame(rows)
    pass_count = int(df.status.eq("PASS").sum())
    fail_count = int(df.status.eq("FAIL").sum())
    no_pass_count = int(df.status.eq("NO_PASS").sum())
    starts = len(df)
    return {
        "portfolio": spec.name,
        "starts": starts,
        "total_trades": total_trades,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "no_pass_count": no_pass_count,
        "pass_rate_pct": round(pass_count / starts * 100.0, 2) if starts else 0.0,
        "fail_rate_pct": round(fail_count / starts * 100.0, 2) if starts else 0.0,
        "median_trades_to_resolution": round(float(df.trades_used.median()), 2) if starts else 0.0,
        "avg_return_pct": round(float(df.net_return_pct.mean()), 3) if starts else 0.0,
        "worst_daily_loss_pct": round(float(df.max_daily_loss_pct.max()), 3) if starts else 0.0,
        "worst_dd_pct": round(float(df.max_dd_pct.min()), 3) if starts else 0.0,
        "max_overall_fails": int(df.fail_reason.eq("max_overall_loss").sum()),
        "max_daily_fails": int(df.fail_reason.eq("max_daily_loss").sum()),
        "legs": "; ".join(f"{leg.strategy}@{leg.risk_pct * 100:.2f}%" for leg in spec.legs),
        "daily_soft_stop_pct": round(spec.daily_soft_stop_pct * 100.0, 3),
        "profit_lock_start_pct": round(spec.profit_lock_start_pct * 100.0, 3),
        "profit_lock_multiplier": spec.profit_lock_multiplier,
        "max_trades_per_day": spec.max_trades_per_day,
    }


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_report(out_dir: Path, summaries: List[Dict[str, object]]) -> str:
    sorted_rows = sorted(summaries, key=lambda r: (r["pass_rate_pct"], -r["fail_rate_pct"], r["avg_return_pct"]), reverse=True)
    lines = [
        "# ProFirm Portfolio Challenge Simulation",
        "",
        "Rules: 10k initial, +5% target, 3% max daily, 5% max overall.",
        "Risk is per-leg fixed risk × R-multiple; momentum_surfer is included in multiple portfolios.",
        "Soft controls: daily soft stop, max trades/day, and profit-lock risk reduction after +X%.",
        "",
        "## Ranked portfolios",
    ]
    for row in sorted_rows:
        lines.append(
            f"- {row['portfolio']}: pass={row['pass_rate_pct']}%, fail={row['fail_rate_pct']}%, "
            f"medianTrades={row['median_trades_to_resolution']}, avgRet={row['avg_return_pct']}%, "
            f"worstDaily={row['worst_daily_loss_pct']}%, worstDD={row['worst_dd_pct']}%, "
            f"legs={row['legs']}"
        )
    lines.extend(
        [
            "",
            "## Files",
            "- `portfolio_summary.csv`: one row per portfolio.",
            "- `portfolio_rolling_details.csv`: every rolling-start simulation.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / stamp
    data = load_all_trades()
    summaries: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []
    for spec in build_portfolios():
        portfolio_trades = combine_portfolio(data, spec)
        if portfolio_trades.empty:
            continue
        rows = [simulate_sequence(portfolio_trades, spec, i) for i in range(len(portfolio_trades))]
        detail_rows.extend(rows)
        summaries.append(summarize(rows, spec, len(portfolio_trades)))
    summaries = sorted(summaries, key=lambda r: (r["pass_rate_pct"], -r["fail_rate_pct"], r["avg_return_pct"]), reverse=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "portfolio_summary.csv", summaries)
    write_csv(out_dir / "portfolio_rolling_details.csv", detail_rows)
    (out_dir / "REPORT.md").write_text(build_report(out_dir, summaries), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps({"out_dir": str(out_dir), "summaries": summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(out_dir), "summaries": summaries[:20]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
