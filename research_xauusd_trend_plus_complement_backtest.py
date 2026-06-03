#!/usr/bin/env python3.14
"""Backtest current XAUUSD trend logic, standalone complement, and combined trend+complement.

Complement = SELL-only false-breakout upthrust overlay in BEAR HTF.
Read-only research script; no MT5 orders and no active_plan edits.
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
OUT_DIR = ROOT / "backtest_reports_trend_plus_complement"
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
    mode: str  # trend_only, complement_only, trend_plus_complement, parallel_sleeves
    risk_pct: float = 0.0035
    max_lots: float = 1.8
    fast_sma: int = 20
    slow_sma: int = 60
    htf_fast_sma: int = 50
    htf_slow_sma: int = 200
    trend_threshold: float = 0.35
    htf_comp_momentum_threshold: float = 1.10
    htf_momentum_bias_weight: float = 0.0
    momentum_score_weight: float = 0.25
    atr_period: int = 14
    breakout_lookback: int = 20
    stop_atr: float = 2.5
    reward_multiple: float = 2.5
    cooldown_bars: int = 2
    max_hold_bars: int = 96
    fb_lookback: int = 20
    fb_min_atr: float = 0.15
    fb_close_back_atr: float = 0.05
    fb_wick_ratio: float = 0.45


@dataclasses.dataclass(frozen=True)
class Snapshot:
    bar_time: dt.datetime
    close: float
    atr: float
    fast_sma: float
    slow_sma: float
    htf_signal: str
    compensated_htf_signal: str
    momentum: float
    m15_momentum: float
    score: float
    primary_signal: str
    complement_signal: str
    final_signal: str
    signal_source: str
    session: str


@dataclasses.dataclass(frozen=True)
class Trade:
    strategy: str
    signal_source: str
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
        tf_map = {"M5": mt5.TIMEFRAME_M5, "H1": mt5.TIMEFRAME_H1, "M15": mt5.TIMEFRAME_M15}
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
        bars.sort(key=lambda bar: bar.time)
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


def htf_context(
    bars: Sequence[Bar], times: Sequence[dt.datetime], ts: dt.datetime, params: Params,
    cache: Dict[int, Tuple[float, float, str]],
) -> Tuple[float, float, str]:
    idx = bisect_right(times, ts.replace(minute=0, second=0, microsecond=0)) - 1
    if idx in cache:
        return cache[idx]
    if idx < params.htf_slow_sma:
        cache[idx] = (0.0, 0.0, "NEUTRAL")
        return cache[idx]
    closes = [bar.close for bar in bars[: idx + 1]]
    fast = mean(closes[-params.htf_fast_sma :])
    slow = mean(closes[-params.htf_slow_sma :])
    recent_slope = closes[-1] - closes[-4] if len(closes) >= 4 else 0.0
    if fast > slow and recent_slope >= 0:
        out = (fast, slow, "BULL")
    elif fast < slow and recent_slope <= 0:
        out = (fast, slow, "BEAR")
    elif fast > slow:
        out = (fast, slow, "BULL")
    elif fast < slow:
        out = (fast, slow, "BEAR")
    else:
        out = (fast, slow, "NEUTRAL")
    cache[idx] = out
    return out


def m15_momentum(
    m15_bars: Sequence[Bar], m15_times: Sequence[dt.datetime], ts: dt.datetime, atr_ref: float,
) -> float:
    idx = bisect_right(m15_times, ts.replace(second=0, microsecond=0)) - 1
    if idx < 4 or atr_ref <= 0:
        return 0.0
    closes = [bar.close for bar in m15_bars[max(0, idx - 20) : idx + 1]]
    if len(closes) < 4:
        return 0.0
    return clamp((closes[-1] - closes[-4]) / atr_ref, -2.0, 2.0)


def compensated_htf_signal(htf_signal: str, m15_mom: float, params: Params) -> str:
    threshold = max(params.htf_comp_momentum_threshold, params.trend_threshold * 2.0)
    if m15_mom >= threshold:
        return "BULL"
    if m15_mom <= -threshold:
        return "BEAR"
    return htf_signal


def trend_score(
    hist: Sequence[Bar], htf_fast: float, htf_slow: float, htf_signal: str,
    momentum: float, params: Params,
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

    trend = 0.0
    if closed.close > fast_sma > slow_sma:
        trend = 0.55
    elif closed.close < fast_sma < slow_sma:
        trend = -0.55
    elif closed.close > slow_sma:
        trend = 0.20
    elif closed.close < slow_sma:
        trend = -0.20

    htf_gap_strength = clamp((htf_fast - htf_slow) / max(atr, POINT * 5), -1.5, 1.5)
    htf_bias = htf_gap_strength * 0.28 + clamp(momentum, -2.0, 2.0) * params.htf_momentum_bias_weight
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
    score = clamp(trend + htf_bias + breakout + clamp(momentum, -1.5, 1.5) * params.momentum_score_weight, -1.5, 1.5)
    return score, atr, fast_sma, slow_sma


def complement_signal(hist: Sequence[Bar], atr: float, htf_signal: str, params: Params) -> str:
    if htf_signal != "BEAR" or atr <= 0 or len(hist) < params.fb_lookback + 1:
        return "NONE"
    last = hist[-1]
    prior = hist[-params.fb_lookback - 1 : -1]
    prior_high = max(bar.high for bar in prior)
    bar_range = max(last.high - last.low, POINT)
    body_high = max(last.open, last.close)
    upper_wick_ratio = (last.high - body_high) / bar_range
    upthrust = (
        last.high >= prior_high + params.fb_min_atr * atr
        and last.close <= prior_high - params.fb_close_back_atr * atr
        and upper_wick_ratio >= params.fb_wick_ratio
    )
    return "SELL" if upthrust else "NONE"


def build_snapshot(
    hist: Sequence[Bar], h1_bars: Sequence[Bar], h1_times: Sequence[dt.datetime],
    h1_cache: Dict[int, Tuple[float, float, str]], m15_bars: Sequence[Bar],
    m15_times: Sequence[dt.datetime], params: Params,
) -> Optional[Snapshot]:
    need = max(params.slow_sma, params.atr_period, params.breakout_lookback, params.fb_lookback) + 5
    if len(hist) < need:
        return None
    closed = hist[-1]
    htf_fast, htf_slow, htf = htf_context(h1_bars, h1_times, closed.time, params, h1_cache)
    atr_pre = atr_from_bars(hist, params.atr_period)
    if atr_pre is None:
        return None
    mom = clamp((hist[-1].close - hist[-4].close) / atr_pre if len(hist) >= 4 else 0.0, -2.0, 2.0)
    m15_mom = m15_momentum(m15_bars, m15_times, closed.time, atr_pre)
    htf_comp = compensated_htf_signal(htf, m15_mom, params)
    score, atr, fast_sma, slow_sma = trend_score(hist, htf_fast, htf_slow, htf_comp, mom, params)
    primary = "NONE"
    if htf_comp == "BULL" and score >= params.trend_threshold:
        primary = "BUY"
    elif htf_comp == "BEAR" and score <= -params.trend_threshold:
        primary = "SELL"
    comp = complement_signal(hist, atr, htf_comp, params)

    final = "NONE"
    source = "none"
    if params.mode == "trend_only":
        final, source = primary, "trend" if primary != "NONE" else "none"
    elif params.mode == "complement_only":
        final, source = comp, "complement" if comp != "NONE" else "none"
    elif params.mode == "trend_plus_complement":
        if primary != "NONE":
            final, source = primary, "trend"
        elif comp != "NONE":
            final, source = comp, "complement"
    elif params.mode == "parallel_sleeves":
        if primary != "NONE" and comp != "NONE":
            final, source = primary, "trend+complement"
        elif primary != "NONE":
            final, source = primary, "trend"
        elif comp != "NONE":
            final, source = comp, "complement"
    else:
        raise ValueError(params.mode)

    return Snapshot(
        bar_time=closed.time,
        close=closed.close,
        atr=atr,
        fast_sma=fast_sma,
        slow_sma=slow_sma,
        htf_signal=htf,
        compensated_htf_signal=htf_comp,
        momentum=mom,
        m15_momentum=m15_mom,
        score=score,
        primary_signal=primary,
        complement_signal=comp,
        final_signal=final,
        signal_source=source,
        session=session_of(closed.time),
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


def backtest(
    m5_bars: Sequence[Bar], h1_bars: Sequence[Bar], m15_bars: Sequence[Bar], params: Params,
) -> Tuple[List[Trade], Dict[str, Any]]:
    equity = INITIAL_EQUITY
    peak = equity
    max_dd = 0.0
    cooldown_until = -1
    position: Optional[Dict[str, Any]] = None
    trades: List[Trade] = []
    snapshots = 0
    signal_counts = Counter()
    h1_times = [bar.time for bar in h1_bars]
    m15_times = [bar.time for bar in m15_bars]
    h1_cache: Dict[int, Tuple[float, float, str]] = {}
    warmup = max(260, params.slow_sma + params.breakout_lookback + params.atr_period + 5)

    for i in range(warmup, len(m5_bars) - 1):
        hist = m5_bars[max(0, i - 300) : i + 1]
        snap = build_snapshot(hist, h1_bars, h1_times, h1_cache, m15_bars, m15_times, params)
        if snap is None:
            continue
        snapshots += 1
        signal_counts[f"primary_{snap.primary_signal}"] += 1
        signal_counts[f"complement_{snap.complement_signal}"] += 1
        signal_counts[f"final_{snap.signal_source}"] += 1
        nxt = m5_bars[i + 1]

        if position is not None:
            side = str(position["side"])
            entry = float(position["entry"])
            sl = float(position["sl"])
            tp = float(position["tp"])
            volume = float(position["volume"])
            exit_price: Optional[float] = None
            reason = ""
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
            bars_held = i - int(position["entry_i"])
            if exit_price is None and bars_held >= params.max_hold_bars:
                exit_price, reason = nxt.close, "TIME"
            if exit_price is not None:
                gross, net, commission, spread_cost = costs(entry, exit_price, volume, side)
                equity += net
                peak = max(peak, equity)
                max_dd = min(max_dd, equity / max(peak, 1e-9) - 1.0)
                risk_amount = abs(entry - float(position["orig_sl"])) * volume * CONTRACT_SIZE
                trades.append(
                    Trade(
                        strategy=params.name,
                        signal_source=str(position["signal_source"]),
                        entry_time=str(position["entry_time"]),
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
                        atr=float(position["atr"]),
                        score=float(position["score"]),
                        htf_signal=str(position["htf_signal"]),
                        session=str(position["session"]),
                    )
                )
                position = None
                cooldown_until = i + params.cooldown_bars

        if position is None and i >= cooldown_until and snap.final_signal in {"BUY", "SELL"}:
            entry = nxt.open
            sl_dist = snap.atr * params.stop_atr
            tp_dist = sl_dist * params.reward_multiple
            sl = entry - sl_dist if snap.final_signal == "BUY" else entry + sl_dist
            tp = entry + tp_dist if snap.final_signal == "BUY" else entry - tp_dist
            volume = position_size(equity, params, entry, sl)
            if volume > 0:
                position = {
                    "side": snap.final_signal,
                    "signal_source": snap.signal_source,
                    "entry": entry,
                    "sl": sl,
                    "orig_sl": sl,
                    "tp": tp,
                    "volume": volume,
                    "entry_time": nxt.time.isoformat(),
                    "entry_i": i + 1,
                    "atr": snap.atr,
                    "score": snap.score,
                    "htf_signal": snap.compensated_htf_signal,
                    "session": snap.session,
                }

    metrics = summarize(trades, equity, max_dd)
    metrics.update({"snapshots": snapshots, **dict(signal_counts)})
    return trades, metrics



def backtest_parallel_sleeves(
    m5_bars: Sequence[Bar], h1_bars: Sequence[Bar], m15_bars: Sequence[Bar], params: Params,
) -> Tuple[List[Trade], Dict[str, Any]]:
    """Backtest trend and complement as independent sleeves on the same account.

    Each sleeve can hold one position independently; both share equity for sizing and DD.
    This models same-account duplicate/overlapping positions better than the simple overlay mode.
    """
    equity = INITIAL_EQUITY
    peak = equity
    max_dd = 0.0
    cooldown_until = {"trend": -1, "complement": -1}
    positions: Dict[str, Optional[Dict[str, Any]]] = {"trend": None, "complement": None}
    trades: List[Trade] = []
    snapshots = 0
    signal_counts = Counter()
    h1_times = [bar.time for bar in h1_bars]
    m15_times = [bar.time for bar in m15_bars]
    h1_cache: Dict[int, Tuple[float, float, str]] = {}
    warmup = max(260, params.slow_sma + params.breakout_lookback + params.atr_period + 5)

    def close_position(source: str, pos: Dict[str, Any], i: int, nxt: Bar) -> Optional[Dict[str, Any]]:
        side = str(pos["side"])
        entry = float(pos["entry"])
        sl = float(pos["sl"])
        tp = float(pos["tp"])
        volume = float(pos["volume"])
        exit_price: Optional[float] = None
        reason = ""
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
        gross, net, commission, spread_cost = costs(entry, exit_price, volume, side)
        risk_amount = abs(entry - float(pos["orig_sl"])) * volume * CONTRACT_SIZE
        return {
            "trade": Trade(
                strategy=params.name,
                signal_source=source,
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
            ),
            "net": net,
        }

    for i in range(warmup, len(m5_bars) - 1):
        hist = m5_bars[max(0, i - 300) : i + 1]
        snap = build_snapshot(hist, h1_bars, h1_times, h1_cache, m15_bars, m15_times, params)
        if snap is None:
            continue
        snapshots += 1
        signal_counts[f"primary_{snap.primary_signal}"] += 1
        signal_counts[f"complement_{snap.complement_signal}"] += 1
        nxt = m5_bars[i + 1]

        for source in ("trend", "complement"):
            pos = positions[source]
            if pos is None:
                continue
            closed = close_position(source, pos, i, nxt)
            if closed is None:
                continue
            trade = closed["trade"]
            equity += float(closed["net"])
            peak = max(peak, equity)
            max_dd = min(max_dd, equity / max(peak, 1e-9) - 1.0)
            trades.append(trade)
            positions[source] = None
            cooldown_until[source] = i + params.cooldown_bars

        entries = [("trend", snap.primary_signal), ("complement", snap.complement_signal)]
        for source, signal in entries:
            if positions[source] is not None or i < cooldown_until[source] or signal not in {"BUY", "SELL"}:
                continue
            entry = nxt.open
            sl_dist = snap.atr * params.stop_atr
            tp_dist = sl_dist * params.reward_multiple
            sl = entry - sl_dist if signal == "BUY" else entry + sl_dist
            tp = entry + tp_dist if signal == "BUY" else entry - tp_dist
            volume = position_size(equity, params, entry, sl)
            if volume <= 0:
                continue
            positions[source] = {
                "side": signal,
                "entry": entry,
                "sl": sl,
                "orig_sl": sl,
                "tp": tp,
                "volume": volume,
                "entry_time": nxt.time.isoformat(),
                "entry_i": i + 1,
                "atr": snap.atr,
                "score": snap.score,
                "htf_signal": snap.compensated_htf_signal,
                "session": snap.session,
            }
            signal_counts[f"opened_{source}"] += 1

    metrics = summarize(trades, equity, max_dd)
    metrics.update({"snapshots": snapshots, **dict(signal_counts)})
    return trades, metrics

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
    m15_bars = fetch_bars("M15", MAX_M5_BARS)
    variants = [
        Params(name="current_trend_only", mode="trend_only"),
        Params(name="complement_fb_sell_only", mode="complement_only"),
        Params(name="current_trend_plus_complement", mode="trend_plus_complement"),
        Params(name="parallel_independent_sleeves", mode="parallel_sleeves"),
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
        "m15_bars": len(m15_bars),
        "assumptions": {
            "current_logic": "M5 SMA/breakout/momentum score gated by H1 SMA50/200 plus M15 compensation",
            "complement": "SELL-only false-breakout upthrust in BEAR compensated HTF; only fills combined strategy when primary signal is NONE",
            "exits": "shared current stop_atr=2.5 and reward_multiple=2.5, max_hold=96 M5 bars for research comparability",
            "costs": {"spread_points": SPREAD_POINTS, "commission_per_lot": COMMISSION_PER_LOT},
            "intrabar_order": "conservative: SL before TP if both touched",
            "live_side_effects": "none",
        },
        "grouped": {},
    }
    for params in variants:
        if params.mode == "parallel_sleeves":
            trades, metrics = backtest_parallel_sleeves(m5_bars, h1_bars, m15_bars, params)
        else:
            trades, metrics = backtest(m5_bars, h1_bars, m15_bars, params)
        summary_rows.append({"strategy": params.name, **metrics})
        all_trades.extend(trades)
        diagnostics["grouped"][params.name] = {
            "signal_source": grouped_metrics(trades, "signal_source"),
            "side": grouped_metrics(trades, "side"),
            "session": grouped_metrics(trades, "session"),
            "months_with_trades": len({trade.entry_time[:7] for trade in trades}),
            "trade_month_counts": dict(Counter(trade.entry_time[:7] for trade in trades)),
        }
    summary_fields = list(dict.fromkeys(key for row in summary_rows for key in row.keys()))
    write_csv(OUT_DIR / "trend_plus_complement_summary.csv", summary_rows, summary_fields)
    write_csv(
        OUT_DIR / "trend_plus_complement_trades.csv",
        [dataclasses.asdict(trade) for trade in all_trades],
        [field.name for field in dataclasses.fields(Trade)],
    )
    with (OUT_DIR / "trend_plus_complement_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)
    print(json.dumps({"summary": summary_rows, "diagnostics": diagnostics, "out_dir": str(OUT_DIR)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
