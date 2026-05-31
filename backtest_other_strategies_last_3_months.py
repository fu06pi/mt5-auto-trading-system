#!/usr/bin/env python3.14
"""Backtest non-Doomsday strategy candidates over the most recent ~3 months.

Artifacts:
- backtest_reports_other_strategies_last_3m/other_strategies_last_3m_summary.csv
- backtest_reports_other_strategies_last_3m/other_strategies_last_3m_trades.csv
- backtest_reports_other_strategies_last_3m/other_strategies_last_3m_diagnostics.json
"""
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from backtest_compare_strategies import (
    AsiaLondonBreakoutStrategy,
    Bar as SignalBar,
    BollingerEdgeSqueezeStrategy,
    MetaRegimeSwitchStrategy,
    MomentumSurferStrategy,
    Trade as SignalTrade,
    TrendStrategy,
    backtest_strategy,
    fetch_mt5_bars as fetch_signal_bars,
    regime_stats,
)
from tide_wave_grid_backtest import (
    BasketTrade,
    TideWaveGridConfig,
    backtest_grid,
    fetch_mt5_bars as fetch_grid_bars,
)

ROOT = Path("/home/chain4655/Documents/Projects/MT5")
OUT_DIR = ROOT / "backtest_reports_other_strategies_last_3m"
INITIAL_EQUITY = 10000.0
POINT = 0.01
CONTRACT_SIZE = 100.0
SPREAD_POINTS = 20.0
COMMISSION_PER_LOT = 7.0
DAYS = 90


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def cost_for_volume(volume: float) -> float:
    spread_cost = SPREAD_POINTS * POINT * CONTRACT_SIZE * volume
    commission = COMMISSION_PER_LOT * volume
    return spread_cost + commission


def month_counts(times: Iterable[dt.datetime]) -> Dict[str, int]:
    return dict(Counter(t.strftime("%Y-%m") for t in times))


def signal_metrics_net(trades: Sequence[SignalTrade]) -> Dict[str, Any]:
    equity = INITIAL_EQUITY
    peak = INITIAL_EQUITY
    max_dd = 0.0
    net_pnls: List[float] = []
    gross_pnls: List[float] = []
    costs: List[float] = []
    for trade in sorted(trades, key=lambda t: t.exit_time):
        cost = cost_for_volume(trade.volume)
        net = trade.pnl - cost
        equity += net
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1.0)
        gross_pnls.append(trade.pnl)
        costs.append(cost)
        net_pnls.append(net)

    wins = [p for p in net_pnls if p > 0]
    losses = [p for p in net_pnls if p <= 0]
    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    return {
        "trades": len(trades),
        "gross_pnl": round(sum(gross_pnls), 2),
        "total_costs": round(sum(costs), 2),
        "net_pnl": round(sum(net_pnls), 2),
        "return_pct": round((equity / INITIAL_EQUITY - 1.0) * 100.0, 2),
        "win_rate_pct": round(len(wins) / len(net_pnls) * 100.0, 2) if net_pnls else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0),
        "max_dd_pct": round(max_dd * 100.0, 2),
        "avg_r": round(statistics.fmean([t.r_multiple for t in trades]), 3) if trades else 0.0,
        "expectancy_usd": round(statistics.fmean(net_pnls), 2) if net_pnls else 0.0,
        "months_with_trades": len({t.entry_time.strftime("%Y-%m") for t in trades}),
        "trade_month_counts": month_counts(t.entry_time for t in trades),
    }


def signal_trade_rows(trades: Sequence[SignalTrade], strategy: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for trade in trades:
        row = dataclasses.asdict(trade)
        cost = cost_for_volume(trade.volume)
        row.update(
            {
                "strategy_family": "single_position_signal",
                "strategy": strategy,
                "timeframe": "M15",
                "entry_time": trade.entry_time.isoformat(),
                "exit_time": trade.exit_time.isoformat(),
                "gross_pnl": round(trade.pnl, 2),
                "commission_plus_spread": round(cost, 2),
                "net_pnl": round(trade.pnl - cost, 2),
                "legs": "",
            }
        )
        rows.append(row)
    return rows


def grid_trade_rows(trades: Sequence[BasketTrade]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for trade in trades:
        rows.append(
            {
                "strategy_family": "tide_wave_grid",
                "strategy": trade.strategy,
                "timeframe": trade.timeframe,
                "entry_time": trade.entry_time.isoformat(),
                "exit_time": trade.exit_time.isoformat(),
                "side": trade.side,
                "entry": trade.avg_entry,
                "exit": trade.exit,
                "sl": "",
                "tp": "",
                "volume": trade.total_volume,
                "pnl": trade.pnl_gross,
                "gross_pnl": trade.pnl_gross,
                "commission_plus_spread": round(trade.commission + trade.spread_cost, 2),
                "net_pnl": trade.pnl_net,
                "r_multiple": "",
                "reason": trade.reason,
                "regime": trade.regime,
                "atr": trade.atr,
                "bars_held": trade.bars_held,
                "legs": trade.legs,
            }
        )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = dt.datetime.now().isoformat(timespec="seconds")

    signal_bars: List[SignalBar] = fetch_signal_bars(months=3)
    signal_strategies = [
        BollingerEdgeSqueezeStrategy(),
        TrendStrategy(),
        AsiaLondonBreakoutStrategy(),
        MomentumSurferStrategy(),
        MetaRegimeSwitchStrategy(),
    ]

    summary_rows: List[Dict[str, Any]] = []
    all_trade_rows: List[Dict[str, Any]] = []
    diagnostics: Dict[str, Any] = {
        "generated_at": generated_at,
        "period_mode": "last_approx_90_days_from_mt5_now_or_terminal_history",
        "cost_assumption": {
            "spread_points": SPREAD_POINTS,
            "commission_per_lot": COMMISSION_PER_LOT,
            "point": POINT,
            "contract_size": CONTRACT_SIZE,
        },
        "signal_backtest": {
            "timeframe": "M15",
            "bars": len(signal_bars),
            "start": signal_bars[0].time.isoformat() if signal_bars else "",
            "end": signal_bars[-1].time.isoformat() if signal_bars else "",
            "note": "Single-position signal strategies use the existing research harness; net metrics deduct estimated spread+commission after trade simulation.",
        },
        "grouped": {},
    }

    for strategy in signal_strategies:
        trades, _gross_metrics, _curve = backtest_strategy(signal_bars, strategy)
        metrics = signal_metrics_net(trades)
        summary_rows.append(
            {
                "strategy_family": "single_position_signal",
                "strategy": strategy.name,
                "timeframe": "M15",
                **{k: json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v for k, v in metrics.items()},
            }
        )
        diagnostics["grouped"][strategy.name] = {"regime": regime_stats(trades)}
        all_trade_rows.extend(signal_trade_rows(trades, strategy.name))

    grid_configs = [
        TideWaveGridConfig(
            name="tide_wave_grid_m15_sprint_companion",
            timeframe="M15",
            max_levels=3,
            base_lot=0.03,
            max_total_lots=0.20,
            grid_step_atr=0.80,
            take_profit_atr=0.35,
            hard_stop_atr=3.8,
            z_entry=1.7,
            rsi_buy=42.0,
            rsi_sell=58.0,
        ),
        TideWaveGridConfig(
            name="tide_wave_grid_m15_selective",
            timeframe="M15",
            grid_step_atr=0.80,
            z_entry=2.00,
            rsi_buy=38.0,
            rsi_sell=62.0,
            take_profit_atr=0.90,
            hard_stop_atr=3.0,
            max_levels=3,
            base_lot=0.02,
            max_total_lots=0.12,
            allow_trend_counter=False,
        ),
        TideWaveGridConfig(
            name="tide_wave_grid_m5_defensive",
            timeframe="M5",
            ema_center=55,
            ema_tide_fast=89,
            ema_tide_slow=233,
            z_window=120,
            grid_step_atr=1.05,
            max_levels=3,
            base_lot=0.02,
            max_total_lots=0.18,
            hard_stop_atr=3.0,
            max_hold_bars=144,
        ),
    ]
    bars_by_tf = {}
    for timeframe in sorted({cfg.timeframe for cfg in grid_configs}):
        bars, actual = fetch_grid_bars(timeframe=timeframe, days=DAYS)
        bars_by_tf[timeframe] = bars
        diagnostics[f"grid_{timeframe}"] = {
            "requested_timeframe": timeframe,
            "actual_timeframe": actual,
            "bars": len(bars),
            "start": bars[0].time.isoformat() if bars else "",
            "end": bars[-1].time.isoformat() if bars else "",
        }

    for cfg in grid_configs:
        trades, metrics = backtest_grid(bars_by_tf[cfg.timeframe], cfg)
        summary_rows.append(
            {
                "strategy_family": "tide_wave_grid",
                "strategy": cfg.name,
                "timeframe": cfg.timeframe,
                "trades": metrics["trades"],
                "gross_pnl": "",
                "total_costs": metrics["total_costs"],
                "net_pnl": metrics["net_pnl"],
                "return_pct": metrics["return_pct"],
                "win_rate_pct": metrics["win_rate_pct"],
                "profit_factor": metrics["profit_factor"],
                "max_dd_pct": metrics["max_dd_pct"],
                "avg_r": "",
                "expectancy_usd": metrics["expectancy_usd"],
                "months_with_trades": len({t.entry_time.strftime("%Y-%m") for t in trades}),
                "trade_month_counts": json.dumps(month_counts(t.entry_time for t in trades), ensure_ascii=False),
                "avg_legs": metrics["avg_legs"],
                "avg_hold_bars": metrics["avg_hold_bars"],
                "open_basket_ignored": metrics["open_basket_ignored"],
            }
        )
        all_trade_rows.extend(grid_trade_rows(trades))

    summary_fields = [
        "strategy_family",
        "strategy",
        "timeframe",
        "trades",
        "gross_pnl",
        "total_costs",
        "net_pnl",
        "return_pct",
        "win_rate_pct",
        "profit_factor",
        "max_dd_pct",
        "avg_r",
        "expectancy_usd",
        "months_with_trades",
        "trade_month_counts",
        "avg_legs",
        "avg_hold_bars",
        "open_basket_ignored",
    ]
    trade_fields = [
        "strategy_family",
        "strategy",
        "timeframe",
        "entry_time",
        "exit_time",
        "side",
        "entry",
        "exit",
        "sl",
        "tp",
        "volume",
        "pnl",
        "gross_pnl",
        "commission_plus_spread",
        "net_pnl",
        "r_multiple",
        "reason",
        "regime",
        "atr",
        "bars_held",
        "legs",
    ]
    write_csv(OUT_DIR / "other_strategies_last_3m_summary.csv", summary_rows, summary_fields)
    write_csv(OUT_DIR / "other_strategies_last_3m_trades.csv", all_trade_rows, trade_fields)
    with (OUT_DIR / "other_strategies_last_3m_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    print(json.dumps({"summary": summary_rows, "diagnostics": diagnostics, "out_dir": str(OUT_DIR)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
