#!/usr/bin/env python3
"""Spike: loss-streak driven hedge helper for XAUUSD trend strategy.

Research-only. Reads cached M5 bars and simulates:
- baseline current trend sleeve
- auxiliary hedge sleeve that activates after trend consecutive losses
- hedge enters opposite to the next primary trend signal while weakness mode is active

No MT5 connection, no orders, no active_plan edits.
"""
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path("/home/chain4655/Documents/Projects/MT5")
sys.path.insert(0, str(ROOT))

from research_xauusd_trend_plus_complement_backtest import (  # noqa: E402
    CONTRACT_SIZE,
    INITIAL_EQUITY,
    Params,
    Trade,
    build_snapshot,
    costs,
    mean,
    stdev,
)
from research_xauusd_trend_plus_complement_long_cache_backtest import (  # noqa: E402
    CSV_PATH,
    aggregate_bars,
    load_m5_csv,
)

OUT_DIR = ROOT / "spikes/002-loss-streak-hedge-helper/results"


@dataclasses.dataclass(frozen=True)
class HedgeParams:
    name: str
    trigger_losses: int
    risk_mult: float
    reward_multiple: float
    max_hold_bars: int
    activation_bars: int
    only_same_side_loss: bool = False


@dataclasses.dataclass
class OpenPosition:
    sleeve: str
    side: str
    entry: float
    sl: float
    orig_sl: float
    tp: float
    volume: float
    entry_time: dt.datetime
    entry_i: int
    atr: float
    score: float
    htf_signal: str
    session: str
    context: str


def round_volume(lots: float) -> float:
    return max(0.0, math.floor(lots / 0.01) * 0.01)


def sized_volume(equity: float, risk_pct: float, max_lots: float, entry: float, sl: float) -> float:
    risk_per_lot = abs(entry - sl) * CONTRACT_SIZE
    if risk_per_lot <= 0:
        return 0.0
    return round_volume(min(max_lots, (equity * risk_pct) / risk_per_lot))


def close_if_needed(pos: OpenPosition, i: int, nxt: Any, max_hold_bars: int) -> Optional[Tuple[Trade, float]]:
    exit_price: Optional[float] = None
    reason = ""
    if pos.side == "BUY":
        if nxt.low <= pos.sl:
            exit_price, reason = pos.sl, "SL"
        elif nxt.high >= pos.tp:
            exit_price, reason = pos.tp, "TP"
    else:
        if nxt.high >= pos.sl:
            exit_price, reason = pos.sl, "SL"
        elif nxt.low <= pos.tp:
            exit_price, reason = pos.tp, "TP"
    bars_held = i - pos.entry_i
    if exit_price is None and bars_held >= max_hold_bars:
        exit_price, reason = nxt.close, "TIME"
    if exit_price is None:
        return None
    gross, net, commission, spread_cost = costs(pos.entry, exit_price, pos.volume, pos.side)
    risk_amount = abs(pos.entry - pos.orig_sl) * pos.volume * CONTRACT_SIZE
    trade = Trade(
        strategy="loss_streak_hedge_helper",
        signal_source=pos.sleeve,
        entry_time=pos.entry_time.isoformat(),
        exit_time=nxt.time.isoformat(),
        side=pos.side,
        entry=pos.entry,
        exit=exit_price,
        sl=pos.sl,
        tp=pos.tp,
        volume=pos.volume,
        gross_pnl=gross,
        net_pnl=net,
        commission=commission,
        spread_cost=spread_cost,
        r_multiple=gross / max(risk_amount, 1e-9),
        reason=reason,
        bars_held=bars_held,
        atr=pos.atr,
        score=pos.score,
        htf_signal=pos.htf_signal,
        session=pos.session,
    )
    return trade, net


def summarize(trades: Sequence[Trade], initial_equity: float = INITIAL_EQUITY) -> Dict[str, Any]:
    equity = initial_equity
    peak = equity
    max_dd = 0.0
    wins = []
    losses = []
    for trade in trades:
        equity += trade.net_pnl
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / max(peak, 1e-9) - 1.0)
        if trade.net_pnl > 0:
            wins.append(trade)
        else:
            losses.append(trade)
    gp = sum(t.net_pnl for t in wins)
    gl = -sum(t.net_pnl for t in losses)
    returns = [t.net_pnl / initial_equity for t in trades]
    return {
        "trades": len(trades),
        "net_pnl": round(equity - initial_equity, 2),
        "return_pct": round((equity / initial_equity - 1.0) * 100.0, 2),
        "win_rate_pct": round(100.0 * len(wins) / len(trades), 2) if trades else 0.0,
        "profit_factor": round(gp / gl, 3) if gl > 0 else (999.0 if gp > 0 else 0.0),
        "max_dd_pct": round(max_dd * 100.0, 2),
        "expectancy_usd": round(mean([t.net_pnl for t in trades]), 2) if trades else 0.0,
        "avg_r": round(mean([t.r_multiple for t in trades]), 3) if trades else 0.0,
        "approx_sharpe_per_trade": round(mean(returns) / max(stdev(returns), 1e-12) * math.sqrt(len(returns)), 3) if len(returns) > 1 else 0.0,
    }


def week_key(value: str) -> str:
    iso = dt.datetime.fromisoformat(value).isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def weekly_stats(trades: Sequence[Trade]) -> Dict[str, Any]:
    grouped: Dict[str, float] = defaultdict(float)
    for trade in trades:
        grouped[week_key(trade.entry_time)] += trade.net_pnl
    values = list(grouped.values())
    if not values:
        return {"active_weeks": 0, "positive_active_week_pct": 0.0, "avg_weekly_return_pct": 0.0, "worst_week_pct": 0.0, "best_week_pct": 0.0}
    return {
        "active_weeks": len(values),
        "positive_active_week_pct": round(100.0 * sum(1 for v in values if v > 0) / len(values), 2),
        "avg_weekly_return_pct": round(statistics.fmean(values) / INITIAL_EQUITY * 100.0, 3),
        "worst_week_pct": round(min(values) / INITIAL_EQUITY * 100.0, 3),
        "best_week_pct": round(max(values) / INITIAL_EQUITY * 100.0, 3),
    }


def simulate(m5_bars: Sequence[Any], h1_bars: Sequence[Any], m15_bars: Sequence[Any], hp: HedgeParams) -> Dict[str, Any]:
    base = Params(name="trend_with_loss_streak_hedge", mode="trend_only")
    h1_times = [bar.time for bar in h1_bars]
    m15_times = [bar.time for bar in m15_bars]
    h1_cache: Dict[int, Tuple[float, float, str]] = {}
    warmup = max(260, base.slow_sma + base.breakout_lookback + base.atr_period + 5)

    trend_pos: Optional[OpenPosition] = None
    hedge_pos: Optional[OpenPosition] = None
    cooldown = {"trend": -1, "hedge": -1}
    equity = INITIAL_EQUITY
    trend_trades: List[Trade] = []
    hedge_trades: List[Trade] = []
    combined_trades: List[Trade] = []
    cons_losses = 0
    last_loss_side: Optional[str] = None
    hedge_active_until = -1
    signal_counts = Counter()

    for i in range(warmup, len(m5_bars) - 1):
        hist = m5_bars[max(0, i - 300) : i + 1]
        snap = build_snapshot(hist, h1_bars, h1_times, h1_cache, m15_bars, m15_times, base)
        if snap is None:
            continue
        nxt = m5_bars[i + 1]

        if trend_pos is not None:
            closed = close_if_needed(trend_pos, i, nxt, base.max_hold_bars)
            if closed is not None:
                trade, net = closed
                trend_trades.append(dataclasses.replace(trade, strategy=hp.name, signal_source="trend"))
                combined_trades.append(dataclasses.replace(trade, strategy=hp.name, signal_source="trend"))
                equity += net
                cooldown["trend"] = i + base.cooldown_bars
                if net <= 0:
                    cons_losses += 1
                    last_loss_side = trend_pos.side
                    if cons_losses >= hp.trigger_losses:
                        hedge_active_until = i + hp.activation_bars
                else:
                    cons_losses = 0
                    last_loss_side = None
                trend_pos = None

        if hedge_pos is not None:
            closed = close_if_needed(hedge_pos, i, nxt, hp.max_hold_bars)
            if closed is not None:
                trade, net = closed
                hedge_trades.append(dataclasses.replace(trade, strategy=hp.name, signal_source="hedge"))
                combined_trades.append(dataclasses.replace(trade, strategy=hp.name, signal_source="hedge"))
                equity += net
                cooldown["hedge"] = i + base.cooldown_bars
                hedge_pos = None

        primary = snap.primary_signal
        if trend_pos is None and i >= cooldown["trend"] and primary in {"BUY", "SELL"}:
            entry = nxt.open
            sl_dist = snap.atr * base.stop_atr
            tp_dist = sl_dist * base.reward_multiple
            sl = entry - sl_dist if primary == "BUY" else entry + sl_dist
            tp = entry + tp_dist if primary == "BUY" else entry - tp_dist
            vol = sized_volume(equity, base.risk_pct, base.max_lots, entry, sl)
            if vol >= 0.01:
                trend_pos = OpenPosition("trend", primary, entry, sl, sl, tp, vol, nxt.time, i + 1, snap.atr, snap.score, snap.compensated_htf_signal, snap.session, "baseline")

        weakness_active = i <= hedge_active_until
        if hp.only_same_side_loss and last_loss_side and primary != last_loss_side:
            weakness_active = False
        if hedge_pos is None and weakness_active and i >= cooldown["hedge"] and primary in {"BUY", "SELL"}:
            side = "SELL" if primary == "BUY" else "BUY"
            entry = nxt.open
            sl_dist = snap.atr * base.stop_atr
            tp_dist = sl_dist * hp.reward_multiple
            sl = entry - sl_dist if side == "BUY" else entry + sl_dist
            tp = entry + tp_dist if side == "BUY" else entry - tp_dist
            vol = sized_volume(equity, base.risk_pct * hp.risk_mult, base.max_lots * hp.risk_mult, entry, sl)
            if vol >= 0.01:
                signal_counts["hedge_opened"] += 1
                hedge_pos = OpenPosition("hedge", side, entry, sl, sl, tp, vol, nxt.time, i + 1, snap.atr, snap.score, snap.compensated_htf_signal, snap.session, f"cons_losses={cons_losses};opposite={primary}")

    combined_trades.sort(key=lambda t: t.exit_time)
    return {
        "params": dataclasses.asdict(hp),
        "trend": summarize(trend_trades),
        "hedge": summarize(hedge_trades),
        "combined": summarize(combined_trades),
        "trend_weekly": weekly_stats(trend_trades),
        "hedge_weekly": weekly_stats(hedge_trades),
        "combined_weekly": weekly_stats(combined_trades),
        "hedge_reason_counts": dict(Counter(t.reason for t in hedge_trades)),
        "hedge_side_counts": dict(Counter(t.side for t in hedge_trades)),
        "signal_counts": dict(signal_counts),
        "trend_trades": trend_trades,
        "hedge_trades": hedge_trades,
        "combined_trades": combined_trades,
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row.keys())) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if fields:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m5 = load_m5_csv(CSV_PATH)
    h1 = aggregate_bars(m5, 60)
    m15 = aggregate_bars(m5, 15)

    # Keep this spike grid intentionally compact: every variant simulates the full
    # 100k-bar cache with the current trend logic, so a broad grid is slow and
    # encourages overfitting. The first block is the naive hedge; the second block
    # tests the better-targeted idea from losing-streak inspection: only hedge when
    # the next signal is the same side as the side that just clustered losses.
    grid: List[HedgeParams] = [
        HedgeParams("naive_ls2_r0.25_rr1.0_act48", 2, 0.25, 1.0, 48, 48, False),
        HedgeParams("naive_ls3_r0.25_rr1.5_act48", 3, 0.25, 1.5, 48, 48, False),
        HedgeParams("naive_ls3_r0.5_rr1.5_act48", 3, 0.5, 1.5, 48, 48, False),
        HedgeParams("same_side_ls4_r0.1_rr2.0_act12", 4, 0.1, 2.0, 48, 12, True),
        HedgeParams("same_side_ls4_r0.25_rr2.0_act12", 4, 0.25, 2.0, 48, 12, True),
        HedgeParams("same_side_ls4_r0.5_rr2.0_act12", 4, 0.5, 2.0, 48, 12, True),
        HedgeParams("same_side_ls4_r0.5_rr1.5_act48", 4, 0.5, 1.5, 48, 48, True),
        HedgeParams("same_side_ls3_r0.1_rr2.0_act12", 3, 0.1, 2.0, 48, 12, True),
    ]

    rows: List[Dict[str, Any]] = []
    best_payload: Optional[Dict[str, Any]] = None
    for hp in grid:
        result = simulate(m5, h1, m15, hp)
        row = {
            **result["params"],
            **{f"trend_{k}": v for k, v in result["trend"].items()},
            **{f"hedge_{k}": v for k, v in result["hedge"].items()},
            **{f"combined_{k}": v for k, v in result["combined"].items()},
            **{f"combined_weekly_{k}": v for k, v in result["combined_weekly"].items()},
            "hedge_trades": result["hedge"]["trades"],
        }
        row["dd_improvement_vs_trend_pct"] = round(row["combined_max_dd_pct"] - row["trend_max_dd_pct"], 2)
        row["return_delta_vs_trend_pct"] = round(row["combined_return_pct"] - row["trend_return_pct"], 2)
        rows.append(row)
        # Prefer lower DD first, then positive return delta, then PF.
        if best_payload is None or (
            row["combined_max_dd_pct"] > best_payload["row"]["combined_max_dd_pct"]
            and row["combined_return_pct"] >= best_payload["row"]["trend_return_pct"] * 0.90
        ):
            best_payload = {"row": row, "result": result}

    # Rank for a hedge-helper candidate, not for a pure risk reducer:
    # prefer positive standalone hedge expectancy and positive return delta, then DD improvement.
    rows.sort(
        key=lambda r: (
            float(r["hedge_return_pct"]) > 0,
            float(r["return_delta_vs_trend_pct"]) > 0,
            float(r["dd_improvement_vs_trend_pct"]),
            float(r["combined_return_pct"]),
            float(r["combined_max_dd_pct"]),
        ),
        reverse=True,
    )
    write_csv(OUT_DIR / "loss_streak_hedge_grid.csv", rows)
    if rows:
        top = rows[0]
        top_result = simulate(m5, h1, m15, HedgeParams(
            name=str(top["name"]),
            trigger_losses=int(top["trigger_losses"]),
            risk_mult=float(top["risk_mult"]),
            reward_multiple=float(top["reward_multiple"]),
            max_hold_bars=int(top["max_hold_bars"]),
            activation_bars=int(top["activation_bars"]),
            only_same_side_loss=str(top["only_same_side_loss"]) == "True",
        ))
        write_csv(OUT_DIR / "best_hedge_trades.csv", [dataclasses.asdict(t) for t in top_result["hedge_trades"]])
        write_csv(OUT_DIR / "best_combined_trades.csv", [dataclasses.asdict(t) for t in top_result["combined_trades"]])

    diagnostics = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "initial_equity": INITIAL_EQUITY,
        "source_csv": str(CSV_PATH),
        "m5_bars": len(m5),
        "m5_start": m5[0].time.isoformat(),
        "m5_end": m5[-1].time.isoformat(),
        "grid_count": len(grid),
        "top_rows": rows[:10],
        "outputs": {
            "grid": str(OUT_DIR / "loss_streak_hedge_grid.csv"),
            "best_hedge_trades": str(OUT_DIR / "best_hedge_trades.csv"),
            "best_combined_trades": str(OUT_DIR / "best_combined_trades.csv"),
        },
    }
    (OUT_DIR / "loss_streak_hedge_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
