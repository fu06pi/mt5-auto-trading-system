#!/usr/bin/env python3.14
"""Backtest inverse/left-side bottom-fishing variants against the current XAUUSD MT5 trend logic.

The current live plan is trend-following: M5 SMA/breakout/momentum score gated by H1 SMA50/200.
This research script tests BUY-only contrarian bottom-fishing entries that fire when the live logic
would be bearish or nearly bearish, instead of enabling anything live.
"""
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import math
import statistics
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from pymt5linux import MetaTrader5
except ImportError:
    sys.path.insert(0, "/home/chain4655/.openharness-venv/lib64/python3.14/site-packages")
    from pymt5linux import MetaTrader5

ROOT = Path("/home/chain4655/Documents/Projects/MT5")
OUT_DIR = ROOT / "backtest_reports_inverse_left_bottom"
SYMBOL = "XAUUSD"
INITIAL_EQUITY = 10000.0
CONTRACT_SIZE = 100.0
POINT = 0.01
SPREAD_POINTS = 20.0
COMMISSION_PER_LOT = 7.0
MAX_M5_BARS = 50000
MAX_H1_BARS = 50000


@dataclasses.dataclass(frozen=True)
class Bar:
    time: dt.datetime
    open: float
    high: float
    low: float
    close: float
    tick_volume: float


@dataclasses.dataclass(frozen=True)
class Params:
    name: str
    mode: str
    risk_pct: float = 0.0035
    max_lots: float = 1.8
    fast_sma: int = 20
    slow_sma: int = 60
    htf_fast_sma: int = 50
    htf_slow_sma: int = 200
    trend_threshold: float = 0.35
    momentum_score_weight: float = 0.25
    atr_period: int = 14
    breakout_lookback: int = 20
    stop_atr: float = 2.5
    reward_multiple: float = 2.5
    cooldown_bars: int = 2
    max_hold_bars: int = 96
    min_down_break_atr: float = 0.10
    close_back_atr: float = 0.05
    min_lower_wick_ratio: float = 0.35
    max_momentum: float = -0.10
    require_bear_htf: bool = True
    require_close_below_slow: bool = True
    allowed_sessions: Optional[Tuple[str, ...]] = None


@dataclasses.dataclass(frozen=True)
class Snapshot:
    bar_time: dt.datetime
    close: float
    atr: float
    fast_sma: float
    slow_sma: float
    htf_fast_sma: float
    htf_slow_sma: float
    htf_signal: str
    momentum: float
    score: float
    live_signal: str
    contrarian_signal: str
    reason: str
    session: str


@dataclasses.dataclass(frozen=True)
class Trade:
    strategy: str
    entry_time: str
    exit_time: str
    side: str
    entry: float
    exit: float
    sl: float
    tp: float
    volume: float
    gross_pnl: float
    net_pnl: float
    commission: float
    spread_cost: float
    r_multiple: float
    reason: str
    bars_held: int
    atr: float
    score: float
    htf_signal: str
    session: str


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def stdev(values: Sequence[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def fetch_bars(timeframe: str, count: int) -> List[Bar]:
    mt5 = MetaTrader5(host="127.0.0.1", port=18812)
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        tf_map = {"M5": mt5.TIMEFRAME_M5, "H1": mt5.TIMEFRAME_H1}
        rates = mt5.copy_rates_from_pos(SYMBOL, tf_map[timeframe], 0, count)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"copy_rates_from_pos({timeframe}) failed: {mt5.last_error()}")
        bars = [
            Bar(
                time=dt.datetime.fromtimestamp(int(row["time"])),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                tick_volume=float(row["tick_volume"]),
            )
            for row in rates
        ]
        bars.sort(key=lambda b: b.time)
        return bars
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


def atr_from_bars(bars: Sequence[Bar], period: int) -> Optional[float]:
    if len(bars) < period + 1:
        return None
    trs = []
    for i in range(-period, 0):
        curr = bars[i]
        prev = bars[i - 1]
        trs.append(max(curr.high - curr.low, abs(curr.high - prev.close), abs(curr.low - prev.close)))
    return max(mean(trs), POINT * 5)


def session_of(ts: dt.datetime) -> str:
    if 0 <= ts.hour < 7:
        return "asia"
    if 7 <= ts.hour < 13:
        return "london_pre_us"
    if 13 <= ts.hour < 20:
        return "us_london_overlap"
    return "late_us"


def h1_context(
    h1_bars: Sequence[Bar],
    h1_times: Sequence[dt.datetime],
    ts: dt.datetime,
    params: Params,
    cache: Dict[int, Tuple[float, float, str]],
) -> Tuple[float, float, str]:
    # Use the last completed H1 bar at or before the M5 closed bar.
    idx = bisect_right(h1_times, ts.replace(minute=0, second=0, microsecond=0)) - 1
    if idx in cache:
        return cache[idx]
    if idx < params.htf_slow_sma:
        cache[idx] = (0.0, 0.0, "NEUTRAL")
        return cache[idx]
    closes = [bar.close for bar in h1_bars[: idx + 1]]
    fast = mean(closes[-params.htf_fast_sma :])
    slow = mean(closes[-params.htf_slow_sma :])
    recent_slope = closes[-1] - closes[-4] if len(closes) >= 4 else 0.0
    if fast > slow and recent_slope >= 0:
        cache[idx] = (fast, slow, "BULL")
        return cache[idx]
    if fast < slow and recent_slope <= 0:
        cache[idx] = (fast, slow, "BEAR")
        return cache[idx]
    if fast > slow:
        cache[idx] = (fast, slow, "BULL")
        return cache[idx]
    if fast < slow:
        cache[idx] = (fast, slow, "BEAR")
        return cache[idx]
    cache[idx] = (fast, slow, "NEUTRAL")
    return cache[idx]


def live_trend_score(
    hist: Sequence[Bar], htf_fast: float, htf_slow: float, htf_signal: str, params: Params
) -> Tuple[float, float, float, float]:
    closed = hist[-1]
    closes = [bar.close for bar in hist]
    highs = [bar.high for bar in hist]
    lows = [bar.low for bar in hist]
    atr = atr_from_bars(hist, params.atr_period)
    if atr is None:
        raise RuntimeError("ATR unavailable")
    fast_sma = mean(closes[-params.fast_sma :])
    slow_sma = mean(closes[-params.slow_sma :])
    momentum = clamp((closes[-1] - closes[-4]) / atr if len(closes) >= 4 else 0.0, -2.0, 2.0)

    trend = 0.0
    if closed.close > fast_sma > slow_sma:
        trend = 0.55
    elif closed.close < fast_sma < slow_sma:
        trend = -0.55
    elif closed.close > slow_sma:
        trend = 0.20
    elif closed.close < slow_sma:
        trend = -0.20

    htf_gap = htf_fast - htf_slow
    htf_gap_strength = clamp(htf_gap / max(atr, POINT * 5), -1.5, 1.5)
    htf_bias = htf_gap_strength * 0.28
    if htf_signal == "BULL":
        htf_bias = max(htf_bias, 0.22 + max(htf_gap_strength, 0.0) * 0.18)
    elif htf_signal == "BEAR":
        htf_bias = min(htf_bias, -0.22 + min(htf_gap_strength, 0.0) * 0.18)
    elif htf_signal == "NEUTRAL":
        htf_bias = 0.0
    if htf_signal == "BULL" and momentum < 0:
        htf_bias *= 0.65
    elif htf_signal == "BEAR" and momentum > 0:
        htf_bias *= 0.65

    lookback = max(10, params.breakout_lookback)
    recent_high = max(highs[-lookback:])
    recent_low = min(lows[-lookback:])
    breakout = clamp((closed.close - recent_high) / atr, -1.0, 1.0) * 0.30
    breakout += clamp((recent_low - closed.close) / atr, -1.0, 1.0) * -0.30
    momentum_component = clamp(momentum, -1.5, 1.5) * params.momentum_score_weight
    score = clamp(trend + htf_bias + breakout + momentum_component, -1.5, 1.5)
    return score, atr, fast_sma, slow_sma


def build_snapshot(
    m5_hist: Sequence[Bar],
    h1_bars: Sequence[Bar],
    h1_times: Sequence[dt.datetime],
    h1_cache: Dict[int, Tuple[float, float, str]],
    params: Params,
) -> Optional[Snapshot]:
    need = max(params.slow_sma, params.atr_period, params.breakout_lookback) + 5
    if len(m5_hist) < need:
        return None
    closed = m5_hist[-1]
    htf_fast, htf_slow, htf_signal = h1_context(h1_bars, h1_times, closed.time, params, h1_cache)
    score, atr, fast_sma, slow_sma = live_trend_score(m5_hist, htf_fast, htf_slow, htf_signal, params)
    live_signal = "SELL" if htf_signal == "BEAR" and score <= -params.trend_threshold else "BUY" if htf_signal == "BULL" and score >= params.trend_threshold else "NONE"

    bars = list(m5_hist)
    last = bars[-1]
    prior = bars[-params.breakout_lookback - 1 : -1]
    prior_low = min(bar.low for bar in prior)
    bar_range = max(last.high - last.low, POINT)
    body_low = min(last.open, last.close)
    lower_wick_ratio = (body_low - last.low) / bar_range
    down_break_atr = (prior_low - last.low) / max(atr, POINT * 5)
    close_back = (last.close - prior_low) / max(atr, POINT * 5)
    below_slow = last.close < slow_sma

    contrarian_signal = "NONE"
    reason = "none"
    if params.mode == "direct_inverse_sell_to_buy":
        if live_signal == "SELL":
            contrarian_signal = "BUY"
            reason = "live_sell_inverse"
    elif params.mode == "spring_bottom":
        checks = [
            (not params.require_bear_htf or htf_signal == "BEAR", "bear_htf"),
            (not params.require_close_below_slow or below_slow, "below_slow"),
            (down_break_atr >= params.min_down_break_atr, "down_break"),
            (close_back >= params.close_back_atr, "close_back"),
            (lower_wick_ratio >= params.min_lower_wick_ratio, "lower_wick"),
            (score <= -params.trend_threshold or live_signal == "SELL", "opposite_trend"),
            (clamp((last.close - bars[-4].close) / atr if len(bars) >= 4 else 0.0, -2.0, 2.0) <= params.max_momentum, "still_left_side"),
        ]
        if all(ok for ok, _ in checks):
            contrarian_signal = "BUY"
            reason = "spring_bottom_against_bear_trend"
        else:
            reason = "miss:" + ",".join(label for ok, label in checks if not ok)
    else:
        raise ValueError(f"Unsupported mode: {params.mode}")

    session = session_of(closed.time)
    if params.allowed_sessions is not None and session not in params.allowed_sessions:
        contrarian_signal = "NONE"
        reason = "session_block"

    return Snapshot(
        bar_time=closed.time,
        close=closed.close,
        atr=atr,
        fast_sma=fast_sma,
        slow_sma=slow_sma,
        htf_fast_sma=htf_fast,
        htf_slow_sma=htf_slow,
        htf_signal=htf_signal,
        momentum=clamp((bars[-1].close - bars[-4].close) / atr if len(bars) >= 4 else 0.0, -2.0, 2.0),
        score=score,
        live_signal=live_signal,
        contrarian_signal=contrarian_signal,
        reason=reason,
        session=session,
    )


def position_size(equity: float, params: Params, entry: float, sl: float) -> float:
    risk = equity * params.risk_pct
    risk_per_lot = abs(entry - sl) * CONTRACT_SIZE
    if risk_per_lot <= 0:
        return 0.0
    lots = risk / risk_per_lot
    return max(0.01, min(params.max_lots, math.floor(lots / 0.01) * 0.01))


def costs(entry: float, exit_price: float, volume: float, side: str) -> Tuple[float, float, float, float]:
    mult = 1.0 if side == "BUY" else -1.0
    gross = (exit_price - entry) * volume * CONTRACT_SIZE * mult
    spread_cost = SPREAD_POINTS * POINT * volume * CONTRACT_SIZE
    commission = volume * COMMISSION_PER_LOT
    return gross, gross - spread_cost - commission, commission, spread_cost


def backtest(m5_bars: Sequence[Bar], h1_bars: Sequence[Bar], params: Params) -> Tuple[List[Trade], Dict[str, Any], List[Snapshot]]:
    equity = INITIAL_EQUITY
    peak = equity
    max_dd = 0.0
    cooldown_until = -1
    position: Optional[Dict[str, Any]] = None
    trades: List[Trade] = []
    snapshots: List[Snapshot] = []
    warmup = max(260, params.slow_sma + params.breakout_lookback + params.atr_period + 5)

    h1_times = [bar.time for bar in h1_bars]
    h1_cache: Dict[int, Tuple[float, float, str]] = {}

    for i in range(warmup, len(m5_bars) - 1):
        hist = m5_bars[max(0, i - 300) : i + 1]
        snapshot = build_snapshot(hist, h1_bars, h1_times, h1_cache, params)
        if snapshot is None:
            continue
        snapshots.append(snapshot)
        nxt = m5_bars[i + 1]

        if position is not None:
            entry = float(position["entry"])
            sl = float(position["sl"])
            tp = float(position["tp"])
            volume = float(position["volume"])
            exit_price: Optional[float] = None
            reason = ""
            if nxt.low <= sl:
                exit_price, reason = sl, "SL"
            elif nxt.high >= tp:
                exit_price, reason = tp, "TP"
            bars_held = i - int(position["entry_i"])
            if exit_price is None and bars_held >= params.max_hold_bars:
                exit_price, reason = nxt.close, "TIME"
            if exit_price is not None:
                gross, net, commission, spread_cost = costs(entry, exit_price, volume, "BUY")
                equity += net
                peak = max(peak, equity)
                max_dd = min(max_dd, equity / max(peak, 1e-9) - 1.0)
                risk_amount = abs(entry - float(position["orig_sl"])) * volume * CONTRACT_SIZE
                trades.append(
                    Trade(
                        strategy=params.name,
                        entry_time=str(position["entry_time"]),
                        exit_time=nxt.time.isoformat(),
                        side="BUY",
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
                        atr=float(position["atr"]),
                        score=float(position["score"]),
                        htf_signal=str(position["htf_signal"]),
                        session=str(position["session"]),
                    )
                )
                position = None
                cooldown_until = i + params.cooldown_bars

        if position is None and i >= cooldown_until and snapshot.contrarian_signal == "BUY":
            entry = nxt.open
            sl_dist = snapshot.atr * params.stop_atr
            tp_dist = sl_dist * params.reward_multiple
            sl = entry - sl_dist
            tp = entry + tp_dist
            volume = position_size(equity, params, entry, sl)
            if volume > 0:
                position = {
                    "entry": entry,
                    "sl": sl,
                    "orig_sl": sl,
                    "tp": tp,
                    "volume": volume,
                    "entry_time": nxt.time.isoformat(),
                    "entry_i": i + 1,
                    "atr": snapshot.atr,
                    "score": snapshot.score,
                    "htf_signal": snapshot.htf_signal,
                    "session": snapshot.session,
                }

    metrics = summarize(trades, equity, max_dd)
    metrics.update(
        {
            "snapshots": len(snapshots),
            "entry_signals": sum(1 for snap in snapshots if snap.contrarian_signal != "NONE"),
            "live_sell_signals": sum(1 for snap in snapshots if snap.live_signal == "SELL"),
            "bear_htf_bars": sum(1 for snap in snapshots if snap.htf_signal == "BEAR"),
        }
    )
    return trades, metrics, snapshots


def summarize(trades: Sequence[Trade], final_equity: float, max_dd: float) -> Dict[str, Any]:
    wins = [trade for trade in trades if trade.net_pnl > 0]
    losses = [trade for trade in trades if trade.net_pnl <= 0]
    gp = sum(trade.net_pnl for trade in wins)
    gl = -sum(trade.net_pnl for trade in losses)
    returns = [trade.net_pnl / INITIAL_EQUITY for trade in trades]
    return {
        "trades": len(trades),
        "net_pnl": round(final_equity - INITIAL_EQUITY, 2),
        "return_pct": round((final_equity / INITIAL_EQUITY - 1.0) * 100.0, 2),
        "win_rate_pct": round(100.0 * len(wins) / len(trades), 2) if trades else 0.0,
        "profit_factor": round(gp / gl, 3) if gl > 0 else (999.0 if gp > 0 else 0.0),
        "max_dd_pct": round(max_dd * 100.0, 2),
        "expectancy_usd": round(mean([trade.net_pnl for trade in trades]), 2) if trades else 0.0,
        "avg_r": round(mean([trade.r_multiple for trade in trades]), 3) if trades else 0.0,
        "total_costs": round(sum(trade.commission + trade.spread_cost for trade in trades), 2),
        "pnl_stdev_usd": round(stdev([trade.net_pnl for trade in trades]), 2),
        "approx_sharpe_per_trade": round(mean(returns) / max(stdev(returns), 1e-12) * math.sqrt(len(returns)), 3) if len(returns) > 1 else 0.0,
    }


def grouped_metrics(trades: Sequence[Trade], key: str) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Trade]] = defaultdict(list)
    for trade in trades:
        groups[str(getattr(trade, key))].append(trade)
    return {group: summarize(rows, INITIAL_EQUITY + sum(t.net_pnl for t in rows), 0.0) for group, rows in sorted(groups.items())}


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m5_bars = fetch_bars("M5", MAX_M5_BARS)
    h1_bars = fetch_bars("H1", MAX_H1_BARS)
    variants = [
        Params(name="direct_inverse_current_sell_buy", mode="direct_inverse_sell_to_buy"),
        Params(name="spring_bottom_base", mode="spring_bottom"),
        Params(name="spring_bottom_strict", mode="spring_bottom", min_down_break_atr=0.20, close_back_atr=0.10, min_lower_wick_ratio=0.45, max_momentum=-0.20),
        Params(name="spring_bottom_loose", mode="spring_bottom", min_down_break_atr=0.05, close_back_atr=0.00, min_lower_wick_ratio=0.25, max_momentum=0.10),
        Params(name="spring_bottom_rr1p5", mode="spring_bottom", reward_multiple=1.5),
        Params(name="spring_bottom_wide_stop", mode="spring_bottom", stop_atr=3.5, reward_multiple=2.0),
        Params(
            name="spring_bottom_wide_stop_london_late",
            mode="spring_bottom",
            stop_atr=3.5,
            reward_multiple=2.0,
            allowed_sessions=("london_pre_us", "late_us"),
        ),
        Params(
            name="spring_bottom_wide_stop_no_us_overlap",
            mode="spring_bottom",
            stop_atr=3.5,
            reward_multiple=2.0,
            allowed_sessions=("asia", "london_pre_us", "late_us"),
        ),
    ]

    summary_rows: List[Dict[str, Any]] = []
    all_trades: List[Trade] = []
    diagnostics: Dict[str, Any] = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "symbol": SYMBOL,
        "m5_bars": len(m5_bars),
        "m5_start": m5_bars[0].time.isoformat(),
        "m5_end": m5_bars[-1].time.isoformat(),
        "h1_bars": len(h1_bars),
        "h1_start": h1_bars[0].time.isoformat(),
        "h1_end": h1_bars[-1].time.isoformat(),
        "assumptions": {
            "strategy": "BUY-only contrarian bottom-fishing against current M5/H1 trend SELL logic",
            "costs": {"spread_points": SPREAD_POINTS, "commission_per_lot": COMMISSION_PER_LOT},
            "intrabar_order": "conservative: SL before TP if both touched",
            "live_side_effects": "none; research-only script",
        },
        "strategy_grouped": {},
    }

    for params in variants:
        trades, metrics, snapshots = backtest(m5_bars, h1_bars, params)
        row = {"strategy": params.name, **metrics}
        summary_rows.append(row)
        all_trades.extend(trades)
        diagnostics["strategy_grouped"][params.name] = {
            "session": grouped_metrics(trades, "session"),
            "htf_signal": grouped_metrics(trades, "htf_signal"),
            "months_with_trades": len({trade.entry_time[:7] for trade in trades}),
            "trade_month_counts": dict(Counter(trade.entry_time[:7] for trade in trades)),
            "snapshot_reason_counts": dict(Counter(snap.reason for snap in snapshots if snap.contrarian_signal != "NONE")),
        }

    write_csv(OUT_DIR / "inverse_left_bottom_summary.csv", summary_rows, list(summary_rows[0].keys()))
    write_csv(
        OUT_DIR / "inverse_left_bottom_trades.csv",
        [dataclasses.asdict(trade) for trade in all_trades],
        [field.name for field in dataclasses.fields(Trade)],
    )
    with (OUT_DIR / "inverse_left_bottom_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)
    print(json.dumps({"summary": summary_rows, "diagnostics": diagnostics, "out_dir": str(OUT_DIR)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
