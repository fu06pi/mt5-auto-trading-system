#!/usr/bin/env python3
from __future__ import annotations

import csv
import dataclasses
import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Type

from backtest_compare_strategies import (
    AsiaLondonBreakoutStrategy,
    Bar,
    BaseStrategy,
    BollingerEdgeSqueezeStrategy,
    DoomsdayStrategy,
    TrendStrategy,
    backtest_strategy,
    fetch_mt5_bars,
)

ROOT = Path("/home/chain4655/Documents/Projects/MT5")
OUT_DIR = ROOT / "backtest_reports" / "meta_component_optimization"
MIN_ALL_TRADES = 4
MIN_TEST_TRADES = 1


def set_attrs(obj: BaseStrategy, params: Dict[str, Any]) -> BaseStrategy:
    for key, value in params.items():
        setattr(obj, key, value)
    return obj


def grid(params: Dict[str, Sequence[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(params.keys())
    for values in itertools.product(*(params[key] for key in keys)):
        yield dict(zip(keys, values))


def metrics_score(metrics: Dict[str, float], min_trades: int) -> float:
    trades = float(metrics.get("trades", 0))
    if trades < min_trades:
        return -1_000_000.0 + trades
    net = float(metrics.get("net_pnl", 0.0))
    pf = float(metrics.get("profit_factor", 0.0))
    dd = abs(float(metrics.get("max_dd_pct", 0.0)))
    expectancy = float(metrics.get("expectancy_usd", 0.0))
    avg_r = float(metrics.get("avg_r", 0.0))
    # Conservative objective: prefer positive test PnL / expectancy, punish drawdown and PF<1.
    pf_bonus = min(pf, 3.0) * 80.0 if pf > 1.0 else (pf - 1.0) * 250.0
    trade_penalty = 0.0 if trades >= min_trades else (min_trades - trades) * 100.0
    return net + expectancy * min(trades, 20.0) * 0.15 + avg_r * 120.0 + pf_bonus - dd * 90.0 - trade_penalty


def evaluate_variant(
    name: str,
    cls: Type[BaseStrategy],
    params: Dict[str, Any],
    train_bars: List[Bar],
    test_bars: List[Bar],
    all_bars: List[Bar],
) -> Dict[str, Any]:
    train = set_attrs(cls(), params)
    test = set_attrs(cls(), params)
    full = set_attrs(cls(), params)
    _train_trades, train_metrics, _ = backtest_strategy(train_bars, train)
    _test_trades, test_metrics, _ = backtest_strategy(test_bars, test)
    _all_trades, all_metrics, _ = backtest_strategy(all_bars, full)
    return {
        "strategy": name,
        "params": json.dumps(params, sort_keys=True),
        "train_trades": train_metrics["trades"],
        "train_net_pnl": train_metrics["net_pnl"],
        "train_pf": train_metrics["profit_factor"],
        "train_dd_pct": train_metrics["max_dd_pct"],
        "test_trades": test_metrics["trades"],
        "test_net_pnl": test_metrics["net_pnl"],
        "test_pf": test_metrics["profit_factor"],
        "test_dd_pct": test_metrics["max_dd_pct"],
        "test_expectancy_usd": test_metrics["expectancy_usd"],
        "test_avg_r": test_metrics["avg_r"],
        "all_trades": all_metrics["trades"],
        "all_net_pnl": all_metrics["net_pnl"],
        "all_return_pct": all_metrics["return_pct"],
        "all_win_rate_pct": all_metrics["win_rate_pct"],
        "all_pf": all_metrics["profit_factor"],
        "all_dd_pct": all_metrics["max_dd_pct"],
        "all_expectancy_usd": all_metrics["expectancy_usd"],
        "all_avg_r": all_metrics["avg_r"],
        "score": round(metrics_score(test_metrics, MIN_TEST_TRADES) * 0.65 + metrics_score(all_metrics, MIN_ALL_TRADES) * 0.35, 3),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bars = fetch_mt5_bars()
    if len(bars) < 1200:
        raise RuntimeError(f"Not enough bars for component optimization: {len(bars)}")
    split = int(len(bars) * 0.65)
    train_bars = bars[:split]
    # include warmup context before test region so indicators can initialize without lookahead.
    test_bars = bars[max(0, split - 260):]

    specs: List[Tuple[str, Type[BaseStrategy], Dict[str, Sequence[Any]]]] = [
        (
            "doomsday",
            DoomsdayStrategy,
            {
                "threshold": [0.58, 0.72],
                "stop_atr": [3.8, 4.6],
                "reward_multiple": [1.6, 2.2],
                "fast": [8],
                "slow": [34, 42],
                "breakout_lookback": [12, 18],
                "high_vol_min_momentum": [0.65, 0.80],
            },
        ),
        (
            "bollinger_edge_squeeze",
            BollingerEdgeSqueezeStrategy,
            {
                "threshold": [0.30, 0.48],
                "stop_atr": [4.5, 5.5],
                "reward_multiple": [1.8, 2.2],
                "bb_period": [20],
                "edge_pct": [0.16, 0.22],
                "squeeze_quantile": [0.25, 0.35],
                "expansion_ratio": [1.12, 1.25],
            },
        ),
        (
            "trend_following",
            TrendStrategy,
            {
                "threshold": [0.55, 0.70, 0.85],
                "stop_atr": [3.0, 3.8],
                "reward_multiple": [2.5, 3.2],
                "fast": [20, 28],
                "slow": [50, 70],
                "breakout_lookback": [20, 30],
                "max_hold_bars": [48],
            },
        ),
        (
            "asia_london_breakout_ep1",
            AsiaLondonBreakoutStrategy,
            {
                "stop_atr": [4.4, 5.4],
                "reward_multiple": [2.4, 3.0],
                "buffer_atr": [0.21, 0.32],
                "min_asia_atr": [0.55, 0.75],
                "max_asia_atr": [3.0],
                "max_hold_bars": [20, 28],
            },
        ),
    ]

    all_rows: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []
    for name, cls, param_grid in specs:
        baseline = evaluate_variant(name, cls, {"variant": "baseline"}, train_bars, test_bars, bars)
        rows: List[Dict[str, Any]] = [baseline]
        for params in grid(param_grid):
            rows.append(evaluate_variant(name, cls, params, train_bars, test_bars, bars))
        rows.sort(key=lambda row: row["score"], reverse=True)
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        all_rows.extend(rows)
        best_rows.append(rows[0])
        with (OUT_DIR / f"{name}_grid.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    best_rows.sort(key=lambda row: row["strategy"])
    with (OUT_DIR / "best_by_component.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(best_rows[0].keys()))
        writer.writeheader()
        writer.writerows(best_rows)

    payload = {
        "bars": len(bars),
        "start": bars[0].time.isoformat(),
        "end": bars[-1].time.isoformat(),
        "train_start": train_bars[0].time.isoformat(),
        "train_end": train_bars[-1].time.isoformat(),
        "test_start": test_bars[0].time.isoformat(),
        "test_end": test_bars[-1].time.isoformat(),
        "out_dir": str(OUT_DIR),
        "best": best_rows,
    }
    (OUT_DIR / "best_by_component.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
