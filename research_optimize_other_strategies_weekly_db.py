#!/usr/bin/env python3.14
"""Weekly DB-backed parameter optimization for MT5 XAUUSD research strategies.

Read/write market bars through a local SQLite cache, then run one-week parameter grids
without touching live active_plan.json or running strategy processes.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import itertools
import json
import math
import sqlite3
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type

try:
    from pymt5linux import MetaTrader5  # type: ignore[import-not-found]
except ImportError:
    sys.path.insert(0, "/home/chain4655/.openharness-venv/lib64/python3.14/site-packages")
    from pymt5linux import MetaTrader5  # type: ignore[import-not-found]

from backtest_compare_strategies import (
    AsiaLondonBreakoutStrategy,
    Bar,
    BaseStrategy,
    BollingerEdgeSqueezeStrategy,
    DoomsdayStrategy,
    MetaRegimeSwitchStrategy,
    MomentumSurferStrategy,
    Trade,
    TrendStrategy,
    backtest_strategy,
)
from tide_wave_grid_backtest import (
    Bar as GridBar,
    TideWaveGridConfig,
    backtest_grid,
)

ROOT = Path("/home/chain4655/Documents/Projects/MT5")
OUT_ROOT = Path("/home/chain4655/Documents/backtest_reports/weekly_db_strategy_optimization")
DB_PATH = ROOT / "research" / "xauusd_market_data.sqlite3"
SYMBOL = "XAUUSD"
TIMEFRAME = "M15"
INITIAL_EQUITY = 10000.0
CONTRACT_SIZE = 100.0
POINT = 0.01
SPREAD_POINTS = 56.0  # observed live spread at status check; conservative vs older 20pt reports
COMMISSION_PER_LOT = 7.0
TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
FETCH_DAYS = 28
SAMPLE_DAYS = 7
WARMUP_BARS = 220


def ensure_schema(con: sqlite3.Connection) -> None:
    con.execute(
        """
        create table if not exists ohlcv (
            symbol text not null,
            timeframe text not null,
            time integer not null,
            iso_time text not null,
            open real not null,
            high real not null,
            low real not null,
            close real not null,
            tick_volume real not null,
            source text not null,
            updated_at text not null,
            primary key(symbol, timeframe, time)
        )
        """
    )
    con.commit()


def fetch_mt5_to_db(fetch_days: int = FETCH_DAYS) -> Dict[str, Any]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    ensure_schema(con)
    mt5 = MetaTrader5(host="127.0.0.1", port=18812)
    ok = mt5.initialize(path=TERMINAL_PATH)
    if not ok:
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        end = dt.datetime.now()
        start = end - dt.timedelta(days=fetch_days)
        rates = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M15, start, end)
        if rates is None or len(rates) == 0:
            fallback_bars = max(3000, int(fetch_days * 96 * 1.2))
            rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, fallback_bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No MT5 rates: {mt5.last_error()}")
        rows = []
        now = dt.datetime.now().isoformat(timespec="seconds")
        for row in rates:
            ts = int(row["time"])
            iso = dt.datetime.fromtimestamp(ts).isoformat()
            rows.append(
                (
                    SYMBOL,
                    TIMEFRAME,
                    ts,
                    iso,
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row["tick_volume"]),
                    "mt5_bridge",
                    now,
                )
            )
        con.executemany(
            """
            insert into ohlcv(symbol,timeframe,time,iso_time,open,high,low,close,tick_volume,source,updated_at)
            values(?,?,?,?,?,?,?,?,?,?,?)
            on conflict(symbol,timeframe,time) do update set
                iso_time=excluded.iso_time,
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                tick_volume=excluded.tick_volume,
                source=excluded.source,
                updated_at=excluded.updated_at
            """,
            rows,
        )
        con.commit()
        return {"fetched_rows": len(rows), "start": rows[0][3], "end": rows[-1][3], "db": str(DB_PATH)}
    finally:
        try:
            mt5.shutdown()
        except RuntimeError:
            pass
        con.close()


def load_bars_from_db() -> List[Bar]:
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        """
        select iso_time, open, high, low, close, tick_volume
        from ohlcv
        where symbol=? and timeframe=?
        order by time
        """,
        (SYMBOL, TIMEFRAME),
    ).fetchall()
    con.close()
    return [
        Bar(dt.datetime.fromisoformat(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]))
        for r in rows
    ]


def cost_for_volume(volume: float) -> float:
    spread_cost = SPREAD_POINTS * POINT * CONTRACT_SIZE * volume
    commission = COMMISSION_PER_LOT * volume
    return spread_cost + commission


def net_metrics(trades: Sequence[Trade]) -> Dict[str, Any]:
    equity = INITIAL_EQUITY
    peak = INITIAL_EQUITY
    max_dd = 0.0
    gross_pnls: List[float] = []
    costs: List[float] = []
    net_pnls: List[float] = []
    for t in sorted(trades, key=lambda x: x.exit_time):
        cost = cost_for_volume(t.volume)
        net = t.pnl - cost
        equity += net
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1.0)
        gross_pnls.append(t.pnl)
        costs.append(cost)
        net_pnls.append(net)
    wins = [p for p in net_pnls if p > 0]
    losses = [p for p in net_pnls if p <= 0]
    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    return {
        "trades": len(net_pnls),
        "gross_pnl": round(sum(gross_pnls), 2),
        "costs": round(sum(costs), 2),
        "net_pnl": round(sum(net_pnls), 2),
        "return_pct": round((equity / INITIAL_EQUITY - 1.0) * 100.0, 3),
        "win_rate_pct": round(len(wins) / len(net_pnls) * 100.0, 2) if net_pnls else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0),
        "max_dd_pct": round(max_dd * 100.0, 3),
        "expectancy_usd": round(statistics.fmean(net_pnls), 2) if net_pnls else 0.0,
    }


def score_row(row: Dict[str, Any]) -> Tuple[float, float, float, int]:
    trades = int(row["trades"])
    pf = float(row["profit_factor"])
    ret = float(row["return_pct"])
    dd = abs(float(row["max_dd_pct"]))
    # Penalize tiny samples and high drawdown. One week is inherently noisy.
    sample_penalty = 0.35 if trades < 2 else 0.0
    score = ret + min(pf, 5.0) * 0.35 - dd * 0.55 - sample_penalty
    return (round(score, 6), ret, -dd, trades)


def make_strategy(cls: Type[BaseStrategy], params: Dict[str, Any]) -> BaseStrategy:
    s = cls()
    for k, v in params.items():
        setattr(s, k, v)
    return s


def param_product(grid: Dict[str, Sequence[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def single_strategy_grids() -> List[Tuple[str, Type[BaseStrategy], Dict[str, Sequence[Any]]]]:
    return [
        (
            "doomsday",
            DoomsdayStrategy,
            {
                "threshold": [0.52, 0.58, 0.64],
                "stop_atr": [3.0, 3.8, 5.0],
                "reward_multiple": [1.4, 1.8, 2.2],
                "high_vol_min_momentum": [0.6, 0.8, 1.0],
            },
        ),
        (
            "bollinger_edge_squeeze",
            BollingerEdgeSqueezeStrategy,
            {
                "threshold": [0.40, 0.48, 0.60],
                "stop_atr": [3.5, 5.5],
                "reward_multiple": [1.6, 2.2, 3.0],
                "squeeze_quantile": [0.20, 0.35],
                "expansion_ratio": [1.10, 1.25],
            },
        ),
        (
            "trend_following",
            TrendStrategy,
            {
                "threshold": [0.65, 0.85, 1.00],
                "fast": [12, 20, 28],
                "slow": [36, 50, 60],
                "stop_atr": [2.0, 2.5, 3.0],
                "reward_multiple": [1.8, 2.5, 3.2],
            },
        ),
        (
            "asia_london_breakout_ep1",
            AsiaLondonBreakoutStrategy,
            {
                "buffer_atr": [0.10, 0.21, 0.35],
                "min_asia_atr": [0.30, 0.55],
                "max_asia_atr": [2.5, 3.5],
                "stop_atr": [2.5, 4.4],
                "reward_multiple": [1.5, 2.4, 3.0],
            },
        ),
        (
            "momentum_surfer",
            MomentumSurferStrategy,
            {
                "threshold": [0.50, 0.65, 0.80],
                "stop_atr": [1.8, 2.5, 3.0],
                "reward_multiple": [2.0, 3.0, 4.0],
                "mom_lookback": [2, 3, 4],
            },
        ),
    ]


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def optimize_single_strategies(bars: List[Bar], sample_start: dt.datetime) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    trades_rows: List[Dict[str, Any]] = []
    for name, cls, grid in single_strategy_grids():
        combos = list(param_product(grid))
        # Avoid invalid trend fast>=slow combinations.
        if name == "trend_following":
            combos = [p for p in combos if int(p["fast"]) < int(p["slow"])]
        for idx, params in enumerate(combos, 1):
            strat = make_strategy(cls, params)
            trades, _gross, _curve = backtest_strategy(bars, strat)
            sample_trades = [t for t in trades if t.entry_time >= sample_start]
            metrics = net_metrics(sample_trades)
            row = {
                "strategy_family": "single_position_signal",
                "strategy": name,
                "param_id": f"{name}_{idx:04d}",
                "params_json": json.dumps(params, sort_keys=True),
                **metrics,
            }
            row["rank_score"] = score_row(row)[0]
            rows.append(row)
            for t in sample_trades:
                tr = dataclasses.asdict(t)
                tr.update({"param_id": row["param_id"], "params_json": row["params_json"], "strategy": name})
                tr["entry_time"] = t.entry_time.isoformat()
                tr["exit_time"] = t.exit_time.isoformat()
                tr["commission_plus_spread"] = round(cost_for_volume(t.volume), 2)
                tr["net_pnl"] = round(t.pnl - cost_for_volume(t.volume), 2)
                trades_rows.append(tr)
    rows.sort(key=score_row, reverse=True)
    return rows, trades_rows


def to_grid_bars(bars: Sequence[Bar]) -> List[GridBar]:
    return [GridBar(b.time, b.open, b.high, b.low, b.close, b.tick_volume) for b in bars]


def optimize_tide_grid(bars: List[Bar], sample_start: dt.datetime) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    trades_rows: List[Dict[str, Any]] = []
    grid_bars = to_grid_bars(bars)
    cfg_id = 0
    for max_levels, step, tp, hard_stop, z_entry, base_lot, max_total in itertools.product(
        [2, 3],
        [0.65, 0.80, 1.05],
        [0.25, 0.35, 0.50],
        [2.8, 3.8, 5.0],
        [0.9, 1.3, 1.7],
        [0.02, 0.03],
        [0.12, 0.20],
    ):
        cfg_id += 1
        cfg = TideWaveGridConfig(
            name=f"tide_weekly_{cfg_id:04d}",
            timeframe="M15",
            max_levels=max_levels,
            base_lot=base_lot,
            max_total_lots=max_total,
            grid_step_atr=step,
            take_profit_atr=tp,
            hard_stop_atr=hard_stop,
            z_entry=z_entry,
            rsi_buy=42.0,
            rsi_sell=58.0,
            max_hold_bars=64,
        )
        trades, metrics = backtest_grid(grid_bars, cfg)
        sample_trades = [t for t in trades if t.entry_time >= sample_start]
        net_pnls = [t.pnl_net for t in sample_trades]
        equity = INITIAL_EQUITY
        peak = equity
        max_dd = 0.0
        for p in net_pnls:
            equity += p
            peak = max(peak, equity)
            max_dd = min(max_dd, equity / peak - 1.0)
        wins = [p for p in net_pnls if p > 0]
        losses = [p for p in net_pnls if p <= 0]
        gp = sum(wins)
        gl = -sum(losses)
        params = dataclasses.asdict(cfg)
        row = {
            "strategy_family": "tide_wave_grid",
            "strategy": "tide_wave_grid",
            "param_id": cfg.name,
            "params_json": json.dumps(params, sort_keys=True),
            "trades": len(sample_trades),
            "gross_pnl": round(sum(t.pnl_gross for t in sample_trades), 2),
            "costs": round(sum(t.commission + t.spread_cost for t in sample_trades), 2),
            "net_pnl": round(sum(net_pnls), 2),
            "return_pct": round((equity / INITIAL_EQUITY - 1.0) * 100.0, 3),
            "win_rate_pct": round(len(wins) / len(net_pnls) * 100.0, 2) if net_pnls else 0.0,
            "profit_factor": round(gp / gl, 3) if gl > 0 else (999.0 if gp > 0 else 0.0),
            "max_dd_pct": round(max_dd * 100.0, 3),
            "expectancy_usd": round(statistics.fmean(net_pnls), 2) if net_pnls else 0.0,
        }
        row["rank_score"] = score_row(row)[0]
        rows.append(row)
        for t in sample_trades:
            trades_rows.append(
                {
                    "strategy": "tide_wave_grid",
                    "param_id": cfg.name,
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat(),
                    "side": t.side,
                    "legs": t.legs,
                    "volume": t.total_volume,
                    "gross_pnl": round(t.pnl_gross, 2),
                    "commission_plus_spread": round(t.commission + t.spread_cost, 2),
                    "net_pnl": round(t.pnl_net, 2),
                    "reason": t.reason,
                    "regime": t.regime,
                    "params_json": row["params_json"],
                }
            )
    rows.sort(key=score_row, reverse=True)
    return rows, trades_rows


def optimize_meta_baseline(bars: List[Bar], sample_start: dt.datetime) -> Dict[str, Any]:
    strat = MetaRegimeSwitchStrategy()
    trades, _gross, _curve = backtest_strategy(bars, strat)
    sample_trades = [t for t in trades if t.entry_time >= sample_start]
    row = {
        "strategy_family": "meta_selector",
        "strategy": "meta_regime_switch_baseline",
        "param_id": "meta_baseline_0001",
        "params_json": json.dumps({"note": "baseline selector; child grids reported separately"}, sort_keys=True),
        **net_metrics(sample_trades),
    }
    row["rank_score"] = score_row(row)[0]
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DB-backed XAUUSD parameter optimization")
    parser.add_argument("--sample-days", type=int, default=SAMPLE_DAYS, help="Optimization sample window in days")
    parser.add_argument("--fetch-days", type=int, default=FETCH_DAYS, help="Days of M15 bars to refresh from MT5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_days = max(1, int(args.sample_days))
    fetch_days = max(sample_days + 5, int(args.fetch_days))
    fetch_info = fetch_mt5_to_db(fetch_days=fetch_days)
    all_bars = load_bars_from_db()
    if len(all_bars) < WARMUP_BARS + 100:
        raise RuntimeError(f"Not enough bars in DB: {len(all_bars)}")
    db_end = all_bars[-1].time
    sample_start = db_end - dt.timedelta(days=sample_days)
    warmup_start = sample_start - dt.timedelta(minutes=15 * WARMUP_BARS)
    bars = [b for b in all_bars if b.time >= warmup_start]
    if len(bars) < WARMUP_BARS + 50:
        raise RuntimeError(f"Not enough warmup+sample bars: {len(bars)}")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    signal_rows, signal_trades = optimize_single_strategies(bars, sample_start)
    tide_rows, tide_trades = optimize_tide_grid(bars, sample_start)
    meta_row = optimize_meta_baseline(bars, sample_start)
    all_rows = signal_rows + tide_rows + [meta_row]
    all_rows.sort(key=score_row, reverse=True)

    write_csv(out_dir / "weekly_param_grid_all.csv", all_rows)
    write_csv(out_dir / "weekly_param_grid_top50.csv", all_rows[:50])
    write_csv(out_dir / "weekly_param_grid_trades.csv", signal_trades + tide_trades)
    summary_by_strategy = []
    for strategy in sorted({r["strategy"] for r in all_rows}):
        group = [r for r in all_rows if r["strategy"] == strategy]
        top = group[0]
        nonzero = [r for r in group if int(r["trades"]) > 0]
        summary_by_strategy.append(
            {
                "strategy": strategy,
                "tested_params": len(group),
                "nonzero_trade_params": len(nonzero),
                "best_param_id": top["param_id"],
                "best_rank_score": top["rank_score"],
                "best_trades": top["trades"],
                "best_net_pnl": top["net_pnl"],
                "best_return_pct": top["return_pct"],
                "best_pf": top["profit_factor"],
                "best_max_dd_pct": top["max_dd_pct"],
                "best_params_json": top["params_json"],
            }
        )
    summary_by_strategy.sort(key=lambda r: (float(r["best_rank_score"]), float(r["best_return_pct"])), reverse=True)
    write_csv(out_dir / "weekly_best_by_strategy.csv", summary_by_strategy)

    diagnostics = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "db_path": str(DB_PATH),
        "fetch_info": fetch_info,
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "all_db_bars": len(all_bars),
        "run_bars": len(bars),
        "warmup_start": warmup_start.isoformat(),
        "sample_start": sample_start.isoformat(),
        "sample_end": db_end.isoformat(),
        "sample_days": sample_days,
        "fetch_days": fetch_days,
        "warmup_bars": WARMUP_BARS,
        "cost_assumption": {
            "spread_points": SPREAD_POINTS,
            "commission_per_lot": COMMISSION_PER_LOT,
            "point": POINT,
            "contract_size": CONTRACT_SIZE,
        },
        "note": "Optimization sample size is configurable. Larger windows reduce one-week regime overfit, but winners still require rolling walk-forward validation before live activation.",
        "files": [
            "weekly_param_grid_all.csv",
            "weekly_param_grid_top50.csv",
            "weekly_best_by_strategy.csv",
            "weekly_param_grid_trades.csv",
            "diagnostics.json",
        ],
    }
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")

    report_lines = [
        "# Weekly DB Strategy Optimization",
        "",
        f"Generated: {diagnostics['generated_at']}",
        f"DB: `{DB_PATH}`",
        f"Sample: `{diagnostics['sample_start']}` → `{diagnostics['sample_end']}`",
        f"Bars used: {len(bars)} M15 bars including {WARMUP_BARS} warmup bars",
        f"Costs: spread={SPREAD_POINTS} points, commission={COMMISSION_PER_LOT}/lot",
        "",
        "## Best by strategy",
    ]
    for row in summary_by_strategy:
        report_lines.append(
            f"- {row['strategy']}: params={row['tested_params']} nonzero={row['nonzero_trade_params']} "
            f"best={row['best_param_id']} trades={row['best_trades']} net={row['best_net_pnl']} "
            f"ret={row['best_return_pct']}% PF={row['best_pf']} DD={row['best_max_dd_pct']}%"
        )
        report_lines.append(f"  - params: `{row['best_params_json']}`")
    report_lines.extend(
        [
            "",
            "## Caveat",
            "- 樣本越大越能降低單週過擬合，但仍只是 in-sample optimization。",
            "- 建議下一步：把 top candidates 跑 rolling walk-forward，再決定 active_plan。",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "top": all_rows[:10], "best_by_strategy": summary_by_strategy}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
