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
from bisect import bisect_left
from collections import Counter, defaultdict, deque
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
INITIAL_EQUITY = 100000.0
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
    chop_gate: str = "none"  # none, conservative_session
    chop_adx_max: float = 18.0
    chop_efficiency_max: float = 0.18
    chop_atr_ratio_max: float = 0.85
    chop_slope_atr_max: float = 1.00
    chop_alternation_min: float = 0.55
    chop_min_score: float = 0.65
    chop_min_points: int = 3
    chop_non_asia_risk_mult: float = 0.25


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


@dataclasses.dataclass(frozen=True)
class ChopState:
    is_chop: bool
    points: int
    reason: str


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def stdev(values: Sequence[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def true_ranges(bars: Sequence[Bar]) -> List[float]:
    out: List[float] = []
    for i in range(1, len(bars)):
        curr = bars[i]
        prev = bars[i - 1]
        out.append(max(curr.high - curr.low, abs(curr.high - prev.close), abs(curr.low - prev.close)))
    return out


def calculate_adx(bars: Sequence[Bar], period: int = 14) -> float:
    if len(bars) < period * 2 + 2:
        return 50.0
    plus_dm: List[float] = []
    minus_dm: List[float] = []
    trs: List[float] = []
    for i in range(1, len(bars)):
        up_move = bars[i].high - bars[i - 1].high
        down_move = bars[i - 1].low - bars[i].low
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
        trs.append(max(bars[i].high - bars[i].low, abs(bars[i].high - bars[i - 1].close), abs(bars[i].low - bars[i - 1].close)))
    dxs: List[float] = []
    for j in range(period, len(trs) + 1):
        tr_sum = sum(trs[j - period:j])
        if tr_sum <= 0:
            continue
        plus_di = 100.0 * sum(plus_dm[j - period:j]) / tr_sum
        minus_di = 100.0 * sum(minus_dm[j - period:j]) / tr_sum
        denom = plus_di + minus_di
        if denom <= 0:
            continue
        dxs.append(100.0 * abs(plus_di - minus_di) / denom)
    return mean(dxs[-period:]) if dxs else 50.0


def range_efficiency(bars: Sequence[Bar], window: int = 48) -> float:
    if len(bars) < window + 1:
        return 1.0
    recent = bars[-window - 1:]
    net_move = abs(recent[-1].close - recent[0].close)
    path = sum(abs(recent[i].close - recent[i - 1].close) for i in range(1, len(recent)))
    return net_move / max(path, POINT)


def atr_ratio(bars: Sequence[Bar], short_period: int = 14, long_period: int = 96) -> float:
    trs = true_ranges(bars)
    if len(trs) < long_period:
        return 1.0
    short_atr = mean(trs[-short_period:])
    long_atr = mean(trs[-long_period:])
    return short_atr / max(long_atr, POINT)


def alternating_signal_rate(signals: Sequence[str]) -> float:
    directional = [sig for sig in signals if sig in {"BUY", "SELL"}]
    if len(directional) < 4:
        return 0.0
    flips = sum(1 for i in range(1, len(directional)) if directional[i] != directional[i - 1])
    return flips / max(len(directional) - 1, 1)


def detect_chop(hist: Sequence[Bar], snap: Snapshot, recent_signals: Sequence[str], params: Params) -> ChopState:
    closes = [bar.close for bar in hist]
    fast = mean(closes[-params.fast_sma:])
    slow = mean(closes[-params.slow_sma:])
    adx = calculate_adx(hist[-120:], 14)
    efficiency = range_efficiency(hist, 48)
    ratio = atr_ratio(hist, 14, 96)
    slope_atr = abs(fast - slow) / max(snap.atr, POINT)
    alternation = alternating_signal_rate(recent_signals[-36:])

    points = 0
    reasons: List[str] = []
    if adx <= params.chop_adx_max:
        points += 1
        reasons.append("adx")
    if efficiency <= params.chop_efficiency_max:
        points += 1
        reasons.append("efficiency")
    if ratio <= params.chop_atr_ratio_max:
        points += 1
        reasons.append("atr_compression")
    if slope_atr <= params.chop_slope_atr_max:
        points += 1
        reasons.append("flat_sma")
    if alternation >= params.chop_alternation_min:
        points += 1
        reasons.append("alternating_signals")
    if abs(float(snap.score)) <= params.chop_min_score:
        points += 1
        reasons.append("weak_score")
    return ChopState(points >= params.chop_min_points, points, "+".join(reasons) or "none")


def trend_risk_multiplier(chop: ChopState, snap: Snapshot, params: Params) -> float:
    if params.chop_gate != "conservative_session" or not chop.is_chop:
        return 1.0
    if snap.session == "asia":
        return 0.0
    return params.chop_non_asia_risk_mult


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
    # Match live strategy behavior: MT5 returns the current forming H1 bar, and
    # live excludes it with htf_bars[:-1].  For a closed M5 bar at ts, only H1
    # bars with open time strictly before the current hour are fully closed.
    hour_floor = ts.replace(minute=0, second=0, microsecond=0)
    idx = bisect_left(times, hour_floor) - 1
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
    # Match live strategy behavior: exclude the current forming M15 bar.
    minute_floor = (ts.minute // 15) * 15
    m15_floor = ts.replace(minute=minute_floor, second=0, microsecond=0)
    idx = bisect_left(m15_times, m15_floor) - 1
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
    recent_signals: deque[str] = deque(maxlen=48)
    warmup = max(260, params.slow_sma + params.breakout_lookback + params.atr_period + 5)

    for i in range(warmup, len(m5_bars) - 1):
        hist = m5_bars[max(0, i - 300) : i + 1]
        snap = build_snapshot(hist, h1_bars, h1_times, h1_cache, m15_bars, m15_times, params)
        if snap is None:
            continue
        snapshots += 1
        recent_signals.append(snap.primary_signal)
        chop = detect_chop(hist, snap, list(recent_signals), params)
        if chop.is_chop:
            signal_counts["chop_bars"] += 1
            signal_counts[f"chop_reason_{chop.reason}"] += 1
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

        entry_signal = snap.final_signal
        entry_source = snap.signal_source
        risk_mult = 1.0
        if entry_source == "trend" and entry_signal in {"BUY", "SELL"}:
            risk_mult = trend_risk_multiplier(chop, snap, params)
            if risk_mult <= 0.0:
                signal_counts["trend_paused_asia_chop"] += 1
                if params.mode == "trend_plus_complement" and snap.complement_signal in {"BUY", "SELL"}:
                    entry_signal = snap.complement_signal
                    entry_source = "complement"
                    risk_mult = 1.0
                    signal_counts["complement_filled_after_trend_pause"] += 1
                else:
                    entry_signal = "NONE"
                    entry_source = "none"
            elif risk_mult < 1.0:
                signal_counts["trend_scaled_non_asia_chop"] += 1
                signal_counts[f"trend_scaled_{snap.session}_chop"] += 1

        if position is None and i >= cooldown_until and entry_signal in {"BUY", "SELL"}:
            entry = nxt.open
            sl_dist = snap.atr * params.stop_atr
            tp_dist = sl_dist * params.reward_multiple
            sl = entry - sl_dist if entry_signal == "BUY" else entry + sl_dist
            tp = entry + tp_dist if entry_signal == "BUY" else entry - tp_dist
            sized_params = dataclasses.replace(
                params,
                risk_pct=params.risk_pct * risk_mult,
                max_lots=params.max_lots * risk_mult,
            )
            volume = position_size(equity, sized_params, entry, sl)
            if volume > 0:
                position = {
                    "side": entry_signal,
                    "signal_source": entry_source,
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
    recent_signals: deque[str] = deque(maxlen=48)
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
        recent_signals.append(snap.primary_signal)
        chop = detect_chop(hist, snap, list(recent_signals), params)
        if chop.is_chop:
            signal_counts["chop_bars"] += 1
            signal_counts[f"chop_reason_{chop.reason}"] += 1
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
            risk_mult = 1.0
            if source == "trend":
                risk_mult = trend_risk_multiplier(chop, snap, params)
                if risk_mult <= 0.0:
                    signal_counts["trend_paused_asia_chop"] += 1
                    continue
                if risk_mult < 1.0:
                    signal_counts["trend_scaled_non_asia_chop"] += 1
                    signal_counts[f"trend_scaled_{snap.session}_chop"] += 1
            entry = nxt.open
            sl_dist = snap.atr * params.stop_atr
            tp_dist = sl_dist * params.reward_multiple
            sl = entry - sl_dist if signal == "BUY" else entry + sl_dist
            tp = entry + tp_dist if signal == "BUY" else entry - tp_dist
            sized_params = dataclasses.replace(
                params,
                risk_pct=params.risk_pct * risk_mult,
                max_lots=params.max_lots * risk_mult,
            )
            volume = position_size(equity, sized_params, entry, sl)
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
        Params(name="current_trend_only_chop_conservative", mode="trend_only", chop_gate="conservative_session"),
        Params(name="complement_fb_sell_only", mode="complement_only"),
        Params(name="current_trend_plus_complement", mode="trend_plus_complement"),
        Params(
            name="current_trend_plus_complement_chop_conservative",
            mode="trend_plus_complement",
            chop_gate="conservative_session",
        ),
        Params(name="parallel_independent_sleeves", mode="parallel_sleeves"),
        Params(
            name="parallel_independent_sleeves_chop_conservative",
            mode="parallel_sleeves",
            chop_gate="conservative_session",
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
