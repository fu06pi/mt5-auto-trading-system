#!/usr/bin/env python3.14
"""Read-only backtest for a copied current XAUUSD portfolio variant.

Variant requested by user:
- Copy current trend + complement sleeve logic from active_plan, research-only.
- RR 1:1.
- Fixed 0.10 lot per entry.
- Portfolio total exposure cap disabled for research.
- Per-trade gross target profit USD 500; RR 1:1 means gross risk USD 500 too.

No MT5 orders and no active_plan edits.
"""
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import math
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import research_xauusd_trend_plus_complement_backtest as base

ROOT = Path("/home/chain4655/Documents/Projects/MT5")
ACTIVE_PLAN = ROOT / "auto_quant/active_plan.json"
OUT_DIR = ROOT / "backtest_reports_fixed_rr1_500"

FIXED_LOT = 0.10
MAX_TOTAL_LOTS = 9999.0
TARGET_PROFIT_USD = 500.0
# XAUUSD contract size in the shared research script is 100 oz per 1.0 lot.
TARGET_PRICE_DISTANCE = TARGET_PROFIT_USD / (FIXED_LOT * base.CONTRACT_SIZE)
DAILY_DD_LIMIT = 0.030
TOTAL_DD_LIMIT = 0.045


def _arg(cmd: Sequence[str], flag: str, default: str) -> str:
    if flag not in cmd:
        return default
    return cmd[cmd.index(flag) + 1]


def load_current_params() -> base.Params:
    """Create research Params from current active_plan without mutating live config."""
    plan = json.loads(ACTIVE_PLAN.read_text())
    main = plan["cmd"]
    comp = plan["complementary"][0]["cmd"]
    return base.Params(
        name="current_copy_rr1_fixed_0_1_tp500_uncapped",
        mode="parallel_sleeves",
        risk_pct=0.0,  # unused: sizing is fixed-lot in this variant
        max_lots=MAX_TOTAL_LOTS,
        fast_sma=int(_arg(main, "--fast-sma", "20")),
        slow_sma=int(_arg(main, "--slow-sma", "60")),
        htf_fast_sma=int(_arg(main, "--htf-fast-sma", "50")),
        htf_slow_sma=int(_arg(main, "--htf-slow-sma", "200")),
        trend_threshold=float(_arg(main, "--trend-threshold", "0.45")),
        htf_comp_momentum_threshold=float(_arg(main, "--htf-comp-momentum-threshold", "1.10")),
        htf_momentum_bias_weight=float(_arg(main, "--htf-momentum-bias-weight", "0.0")),
        momentum_score_weight=float(_arg(main, "--momentum-score-weight", "0.25")),
        atr_period=int(_arg(main, "--atr-period", "14")),
        breakout_lookback=int(_arg(main, "--breakout-lookback", "20")),
        # Exits are overridden to fixed USD TP/SL, but record RR intent here.
        stop_atr=float(_arg(main, "--stop-atr", "2.5")),
        reward_multiple=1.0,
        cooldown_bars=int(_arg(main, "--cooldown-bars-after-trade", "2")),
        max_hold_bars=max(0, int(int(_arg(main, "--max-hold-minutes", "480")) / 5)),
        fb_lookback=int(_arg(comp, "--false-breakout-lookback", "20")),
        fb_min_atr=float(_arg(comp, "--false-breakout-min-atr", "0.15")),
        fb_close_back_atr=float(_arg(comp, "--false-breakout-close-back-atr", "0.05")),
        fb_wick_ratio=float(_arg(comp, "--false-breakout-wick-ratio", "0.45")),
        chop_gate=str(_arg(main, "--chop-gate", "none")),
        chop_adx_max=float(_arg(main, "--chop-adx-max", "18.0")),
        chop_efficiency_max=float(_arg(main, "--chop-efficiency-max", "0.18")),
        chop_atr_ratio_max=float(_arg(main, "--chop-atr-ratio-max", "0.85")),
        chop_slope_atr_max=float(_arg(main, "--chop-slope-atr-max", "1.00")),
        chop_alternation_min=float(_arg(main, "--chop-alternation-min", "0.55")),
        chop_min_score=float(_arg(main, "--chop-min-score", "0.65")),
        chop_min_points=int(_arg(main, "--chop-min-points", "3")),
        chop_non_asia_risk_mult=float(_arg(main, "--chop-non-asia-risk-mult", "0.88")),
    )


def complement_signal_both(hist: Sequence[base.Bar], atr: float, htf_signal: str, params: base.Params) -> str:
    """Mirror live false-breakout BOTH: SELL upthrust in BEAR, BUY spring in BULL."""
    if atr <= 0 or len(hist) < params.fb_lookback + 1:
        return "NONE"
    last = hist[-1]
    prior = hist[-params.fb_lookback - 1 : -1]
    prior_high = max(bar.high for bar in prior)
    prior_low = min(bar.low for bar in prior)
    bar_range = max(last.high - last.low, base.POINT)
    body_high = max(last.open, last.close)
    body_low = min(last.open, last.close)
    upper_wick_ratio = (last.high - body_high) / bar_range
    lower_wick_ratio = (body_low - last.low) / bar_range
    min_break = params.fb_min_atr * atr
    close_back = params.fb_close_back_atr * atr
    min_wick = params.fb_wick_ratio
    upthrust = (
        last.high >= prior_high + min_break
        and last.close <= prior_high - close_back
        and upper_wick_ratio >= min_wick
    )
    if upthrust and htf_signal == "BEAR":
        return "SELL"
    spring = (
        last.low <= prior_low - min_break
        and last.close >= prior_low + close_back
        and lower_wick_ratio >= min_wick
    )
    if spring and htf_signal == "BULL":
        return "BUY"
    return "NONE"


def build_signals(
    hist: Sequence[base.Bar],
    h1_bars: Sequence[base.Bar],
    h1_times: Sequence[dt.datetime],
    h1_cache: Dict[int, Tuple[float, float, str]],
    m15_bars: Sequence[base.Bar],
    m15_times: Sequence[dt.datetime],
    params: base.Params,
) -> Optional[Tuple[base.Snapshot, str, str]]:
    snap = base.build_snapshot(hist, h1_bars, h1_times, h1_cache, m15_bars, m15_times, params)
    if snap is None:
        return None
    comp = complement_signal_both(hist, snap.atr, snap.compensated_htf_signal, params)
    snap = dataclasses.replace(snap, complement_signal=comp)
    return snap, snap.primary_signal, comp


def make_tp_sl(entry: float, side: str) -> Tuple[float, float]:
    if side == "BUY":
        return entry - TARGET_PRICE_DISTANCE, entry + TARGET_PRICE_DISTANCE
    return entry + TARGET_PRICE_DISTANCE, entry - TARGET_PRICE_DISTANCE


def floating_equity(cash_equity: float, positions: Sequence[Dict[str, Any]], mark_price: float) -> float:
    """Cash plus open-position PnL marked at bar close, including exit costs."""
    equity = cash_equity
    for pos in positions:
        gross, net, _, _ = base.costs(
            float(pos["entry"]),
            mark_price,
            float(pos["volume"]),
            str(pos["side"]),
        )
        equity += net
    return equity


def force_close_trade(pos: Dict[str, Any], bar: base.Bar, reason: str) -> Tuple[base.Trade, float]:
    side = str(pos["side"])
    entry = float(pos["entry"])
    volume = float(pos["volume"])
    exit_price = float(bar.close)
    gross, net, commission, spread_cost = base.costs(entry, exit_price, volume, side)
    trade = base.Trade(
        strategy=str(pos.get("strategy", "current_copy_rr1_fixed_0_1_tp500")),
        signal_source=str(pos["source"]),
        entry_time=str(pos["entry_time"]),
        exit_time=bar.time.isoformat(),
        side=side,
        entry=entry,
        exit=exit_price,
        sl=float(pos["sl"]),
        tp=float(pos["tp"]),
        volume=volume,
        gross_pnl=gross,
        net_pnl=net,
        commission=commission,
        spread_cost=spread_cost,
        r_multiple=gross / max(TARGET_PROFIT_USD, 1e-9),
        reason=reason,
        bars_held=max(0, int(pos.get("last_i", pos["entry_i"])) - int(pos["entry_i"])),
        atr=float(pos["atr"]),
        score=float(pos["score"]),
        htf_signal=str(pos["htf_signal"]),
        session=str(pos["session"]),
    )
    return trade, net


def close_trade(pos: Dict[str, Any], i: int, nxt: base.Bar, params: base.Params) -> Optional[Tuple[base.Trade, float]]:
    side = str(pos["side"])
    entry = float(pos["entry"])
    sl = float(pos["sl"])
    tp = float(pos["tp"])
    volume = float(pos["volume"])
    exit_price: Optional[float] = None
    reason = ""
    # Keep conservative intrabar ordering from the shared script: SL before TP if both touched.
    if side == "BUY":
        if nxt.low <= sl:
            exit_price, reason = sl, "SL"
        elif nxt.high >= tp:
            exit_price, reason = tp, "TP"
    else:
        if nxt.high >= sl:
            exit_price, reason = sl, "SL"
        elif nxt.low <= tp:
            exit_price, reason = tp, "TP"
    bars_held = i - int(pos["entry_i"])
    if exit_price is None and bars_held >= params.max_hold_bars:
        exit_price, reason = nxt.close, "TIME"
    if exit_price is None:
        return None
    gross, net, commission, spread_cost = base.costs(entry, exit_price, volume, side)
    risk_amount = TARGET_PROFIT_USD
    trade = base.Trade(
        strategy="current_copy_rr1_fixed_0_1_tp500_uncapped",
        signal_source=str(pos["source"]),
        entry_time=str(pos["entry_time"]),
        exit_time=nxt.time.isoformat(),
        side=side,
        entry=entry,
        exit=exit_price,
        sl=sl,
        tp=tp,
        volume=volume,
        gross_pnl=gross,
        net_pnl=net,
        commission=commission,
        spread_cost=spread_cost,
        r_multiple=gross / max(risk_amount, 1e-9),
        reason=reason,
        bars_held=bars_held,
        atr=float(pos["atr"]),
        score=float(pos["score"]),
        htf_signal=str(pos["htf_signal"]),
        session=str(pos["session"]),
    )
    return trade, net


def backtest_fixed(m5_bars: Sequence[base.Bar], h1_bars: Sequence[base.Bar], m15_bars: Sequence[base.Bar], params: base.Params) -> Tuple[List[base.Trade], Dict[str, Any]]:
    equity = base.INITIAL_EQUITY
    peak = equity
    max_dd = 0.0
    max_equity_seen = equity
    day_start_equity = equity
    current_day: Optional[dt.date] = None
    stopped_reason = ""
    stopped_at = ""
    positions: List[Dict[str, Any]] = []
    cooldown_until = {"trend": -1, "complement": -1}
    trades: List[base.Trade] = []
    signal_counts = Counter()
    h1_times = [bar.time for bar in h1_bars]
    m15_times = [bar.time for bar in m15_bars]
    h1_cache: Dict[int, Tuple[float, float, str]] = {}
    recent_signals: deque[str] = deque(maxlen=48)
    warmup = max(260, params.slow_sma + params.breakout_lookback + params.atr_period + 5)

    for i in range(warmup, len(m5_bars) - 1):
        hist = m5_bars[max(0, i - 300) : i + 1]
        result = build_signals(hist, h1_bars, h1_times, h1_cache, m15_bars, m15_times, params)
        if result is None:
            continue
        snap, primary, comp = result
        recent_signals.append(primary)
        chop = base.detect_chop(hist, snap, list(recent_signals), params)
        signal_counts[f"primary_{primary}"] += 1
        signal_counts[f"complement_{comp}"] += 1
        if chop.is_chop:
            signal_counts["chop_bars"] += 1
            signal_counts[f"chop_reason_{chop.reason}"] += 1
        nxt = m5_bars[i + 1]

        mark_bar = hist[-1]
        mark_equity = floating_equity(equity, positions, mark_bar.close)
        if current_day != mark_bar.time.date():
            current_day = mark_bar.time.date()
            day_start_equity = mark_equity
        peak = max(peak, mark_equity)
        max_equity_seen = max(max_equity_seen, mark_equity)
        max_dd = min(max_dd, mark_equity / max(peak, 1e-9) - 1.0)
        daily_dd = 1.0 - mark_equity / max(day_start_equity, 1e-9)
        total_dd = 1.0 - mark_equity / max(max_equity_seen, 1e-9)
        if daily_dd >= DAILY_DD_LIMIT or total_dd >= TOTAL_DD_LIMIT:
            stopped_reason = "DAILY_DD_STOP" if daily_dd >= DAILY_DD_LIMIT else "TOTAL_DD_STOP"
            stopped_at = mark_bar.time.isoformat()
            for pos in positions:
                pos["last_i"] = i
                trade, net = force_close_trade(pos, mark_bar, stopped_reason)
                equity += net
                trades.append(trade)
                signal_counts[f"closed_{trade.reason}"] += 1
            positions = []
            signal_counts[stopped_reason.lower()] += 1
            break

        still_open: List[Dict[str, Any]] = []
        for pos in positions:
            closed = close_trade(pos, i, nxt, params)
            if closed is None:
                still_open.append(pos)
                continue
            trade, net = closed
            equity += net
            peak = max(peak, equity)
            max_dd = min(max_dd, equity / max(peak, 1e-9) - 1.0)
            trades.append(trade)
            cooldown_until[str(pos["source"])] = i + params.cooldown_bars
            signal_counts[f"closed_{trade.reason}"] += 1
        positions = still_open

        total_open_lots = sum(float(pos["volume"]) for pos in positions)
        free_lots = round(MAX_TOTAL_LOTS - total_open_lots, 2)
        if free_lots + 1e-9 < FIXED_LOT:
            signal_counts["blocked_total_lot_cap"] += 1
            continue

        candidates = [("trend", primary), ("complement", comp)]
        for source, signal in candidates:
            if signal not in {"BUY", "SELL"} or i < cooldown_until[source]:
                continue
            if source == "trend":
                risk_mult = base.trend_risk_multiplier(chop, snap, params)
                if risk_mult <= 0.0:
                    signal_counts["trend_paused_asia_chop"] += 1
                    continue
            total_open_lots = sum(float(pos["volume"]) for pos in positions)
            if total_open_lots + FIXED_LOT > MAX_TOTAL_LOTS + 1e-9:
                signal_counts["blocked_total_lot_cap"] += 1
                continue
            entry = nxt.open
            sl, tp = make_tp_sl(entry, signal)
            positions.append(
                {
                    "strategy": params.name,
                    "source": source,
                    "side": signal,
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "volume": FIXED_LOT,
                    "entry_time": nxt.time.isoformat(),
                    "entry_i": i + 1,
                    "atr": snap.atr,
                    "score": snap.score,
                    "htf_signal": snap.compensated_htf_signal,
                    "session": snap.session,
                }
            )
            signal_counts[f"opened_{source}"] += 1

    metrics = base.summarize(trades, equity, max_dd)
    metrics.update(
        {
            "open_positions_end": len(positions),
            "open_lots_end": round(sum(float(pos["volume"]) for pos in positions), 2),
            "stopped_reason": stopped_reason or "NONE",
            "stopped_at": stopped_at,
            "daily_dd_limit_pct": round(DAILY_DD_LIMIT * 100.0, 2),
            "total_dd_limit_pct": round(TOTAL_DD_LIMIT * 100.0, 2),
            "target_price_distance": round(TARGET_PRICE_DISTANCE, 2),
            "fixed_lot": FIXED_LOT,
            "max_total_lots": "unlimited",
            **dict(signal_counts),
        }
    )
    return trades, metrics


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    params = load_current_params()
    m5_bars = base.fetch_bars("M5", base.MAX_M5_BARS)
    h1_bars = base.fetch_bars("H1", base.MAX_H1_BARS)
    m15_bars = base.fetch_bars("M15", base.MAX_M5_BARS)
    trades, metrics = backtest_fixed(m5_bars, h1_bars, m15_bars, params)
    summary = {
        "strategy": params.name,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "m5_start": m5_bars[0].time.isoformat(),
        "m5_end": m5_bars[-1].time.isoformat(),
        "m5_bars": len(m5_bars),
        "h1_bars": len(h1_bars),
        "m15_bars": len(m15_bars),
        "assumption_tp_sl": f"gross TP/SL USD {TARGET_PROFIT_USD:.0f}; price distance {TARGET_PRICE_DISTANCE:.2f}; costs deducted separately",
        "live_side_effects": "none",
        **metrics,
    }
    summary_path = OUT_DIR / "current_copy_rr1_fixed_0_1_tp500_uncapped_summary.csv"
    trades_path = OUT_DIR / "current_copy_rr1_fixed_0_1_tp500_uncapped_trades.csv"
    json_path = OUT_DIR / "current_copy_rr1_fixed_0_1_tp500_uncapped_diagnostics.json"
    write_csv(summary_path, [summary], list(summary.keys()))
    trade_rows = [dataclasses.asdict(trade) for trade in trades]
    if trade_rows:
        write_csv(trades_path, trade_rows, list(trade_rows[0].keys()))
    else:
        write_csv(trades_path, [], [field.name for field in dataclasses.fields(base.Trade)])
    grouped = {
        "signal_source": base.grouped_metrics(trades, "signal_source"),
        "side": base.grouped_metrics(trades, "side"),
        "session": base.grouped_metrics(trades, "session"),
        "exit_reason": base.grouped_metrics(trades, "reason"),
    }
    json_path.write_text(
        json.dumps({"summary": summary, "grouped": grouped}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary, "summary_path": str(summary_path), "trades_path": str(trades_path), "json_path": str(json_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
