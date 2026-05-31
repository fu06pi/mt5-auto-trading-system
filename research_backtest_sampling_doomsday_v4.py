#!/usr/bin/env python3.14
"""Doomsday V4 backtest comparison with sampling-representativeness checks.

Purpose:
- Compare the active Doomsday V4 sprint parameters against nearby variants.
- Check whether the available M5 sample is representative enough of the broader XAUUSD regime.
- Emit CSV/JSON artifacts for review.
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from pymt5linux import MetaTrader5
except ImportError:
    sys.path.insert(0, "/home/chain4655/.openharness-venv/lib64/python3.14/site-packages")
    from pymt5linux import MetaTrader5

ROOT = Path("/home/chain4655/Documents/Projects/MT5")
OUT_DIR = ROOT / "backtest_reports_sampling"
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
    risk_pct: float = 0.0075
    stop_atr: float = 3.2
    reward_multiple: float = 2.2
    long_bias: float = 0.70
    trend_threshold: float = 0.55
    cooldown_minutes: int = 25
    atr_period: int = 12
    fast_sma: int = 5
    slow_sma: int = 34
    high_vol_only: bool = True
    high_vol_atr_pct: float = 0.0014
    high_vol_range_atr: float = 4.0
    high_vol_breakout_lookback: int = 14
    high_vol_min_momentum: float = 0.65
    high_vol_spike_atr: float = 2.2
    high_vol_min_breakout_atr: float = 0.20
    high_vol_min_close_location: float = 0.60
    max_lots: float = 1.2


@dataclasses.dataclass(frozen=True)
class Snapshot:
    bar_time: dt.datetime
    close: float
    atr: float
    fast_sma: float
    slow_sma: float
    momentum: float
    score: float
    signal: str
    raw_signal: str
    atr_pct: float
    range_atr: float
    spike_atr: float
    breakout_atr: float
    close_location: float
    high_volatility: bool
    entry_ok: bool
    vol_bucket: str
    session: str


@dataclasses.dataclass(frozen=True)
class Trade:
    strategy: str
    fold: str
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
    atr_pct: float
    vol_bucket: str
    session: str


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * clamp(q, 0.0, 1.0)))
    return ordered[idx]


def stdev(values: Sequence[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def atr_from_bars(bars: Sequence[Bar], period: int) -> Optional[float]:
    if len(bars) < period + 1:
        return None
    trs: List[float] = []
    for i in range(-period, 0):
        curr = bars[i]
        prev = bars[i - 1]
        trs.append(max(curr.high - curr.low, abs(curr.high - prev.close), abs(curr.low - prev.close)))
    return max(mean(trs), POINT * 5)


def session_of(ts: dt.datetime) -> str:
    hour = ts.hour
    if 0 <= hour < 7:
        return "asia"
    if 7 <= hour < 13:
        return "london_pre_us"
    if 13 <= hour < 20:
        return "us_london_overlap"
    return "late_us"


def month_key(ts: dt.datetime) -> str:
    return ts.strftime("%Y-%m")


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


def build_snapshot(hist: Sequence[Bar], params: Params, vol_bucket_thresholds: Tuple[float, float]) -> Optional[Snapshot]:
    need = max(params.slow_sma, params.atr_period) + params.high_vol_breakout_lookback + 5
    if len(hist) < need:
        return None
    closed = hist[-1]
    closes = [bar.close for bar in hist]
    highs = [bar.high for bar in hist]
    lows = [bar.low for bar in hist]
    atr = atr_from_bars(hist, params.atr_period)
    if atr is None or atr <= 0:
        return None
    fast_sma = mean(closes[-params.fast_sma :])
    slow_sma = mean(closes[-params.slow_sma :])
    momentum = clamp((closes[-1] - closes[-4]) / atr if len(closes) >= 4 else 0.0, -2.0, 2.0)
    atr_pct = atr / max(closed.close, 1e-9)
    regime_lookback = min(20, len(highs), len(lows))
    range_atr = (max(highs[-regime_lookback:]) - min(lows[-regime_lookback:])) / max(atr, 1e-9)
    last_range = max(closed.high - closed.low, 0.0)
    spike_atr = last_range / max(atr, 1e-9)
    close_location = (closed.close - closed.low) / last_range if last_range > 0 else 0.5
    breakout_lookback = max(5, min(params.high_vol_breakout_lookback, len(highs) - 1))
    recent_high = max(highs[-breakout_lookback - 1 : -1])
    recent_low = min(lows[-breakout_lookback - 1 : -1])
    if closed.close > recent_high:
        breakout_atr = (closed.close - recent_high) / atr
    elif closed.close < recent_low:
        breakout_atr = (closed.close - recent_low) / atr
    else:
        breakout_atr = 0.0

    trend = 0.0
    if closed.close > fast_sma > slow_sma:
        trend = 0.40
    elif closed.close < fast_sma < slow_sma:
        trend = -0.40
    elif closed.close > slow_sma:
        trend = 0.12
    elif closed.close < slow_sma:
        trend = -0.12
    up_break = clamp((closed.close - recent_high) / atr, -1.5, 1.5)
    dn_break = clamp((recent_low - closed.close) / atr, -1.5, 1.5)
    score = clamp(trend + up_break * 0.45 + dn_break * -0.45 + momentum * 0.45 + (params.long_bias - 0.5) * 0.20, -1.5, 1.5)
    raw_signal = "BUY" if score >= params.trend_threshold else "SELL" if score <= -params.trend_threshold else "NONE"
    high_volatility = atr_pct >= params.high_vol_atr_pct and (range_atr >= params.high_vol_range_atr or spike_atr >= params.high_vol_spike_atr)
    signal = raw_signal
    if params.high_vol_only and not high_volatility:
        signal = "NONE"
    entry_ok = False
    if signal == "BUY":
        entry_ok = (
            high_volatility
            and momentum > 0
            and abs(momentum) >= params.high_vol_min_momentum
            and breakout_atr >= params.high_vol_min_breakout_atr
            and close_location >= params.high_vol_min_close_location
        )
    elif signal == "SELL":
        entry_ok = (
            high_volatility
            and momentum < 0
            and abs(momentum) >= params.high_vol_min_momentum
            and breakout_atr <= -params.high_vol_min_breakout_atr
            and close_location <= 1.0 - params.high_vol_min_close_location
        )
    if not params.high_vol_only and raw_signal != "NONE":
        entry_ok = True
        signal = raw_signal
    if signal != "NONE" and not entry_ok:
        signal = "NONE"

    p33, p67 = vol_bucket_thresholds
    if atr_pct <= p33:
        vol_bucket = "low"
    elif atr_pct <= p67:
        vol_bucket = "mid"
    else:
        vol_bucket = "high"

    return Snapshot(
        bar_time=closed.time,
        close=closed.close,
        atr=atr,
        fast_sma=fast_sma,
        slow_sma=slow_sma,
        momentum=momentum,
        score=score,
        signal=signal,
        raw_signal=raw_signal,
        atr_pct=atr_pct,
        range_atr=range_atr,
        spike_atr=spike_atr,
        breakout_atr=breakout_atr,
        close_location=close_location,
        high_volatility=high_volatility,
        entry_ok=entry_ok,
        vol_bucket=vol_bucket,
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


def fold_for_time(ts: dt.datetime, start: dt.datetime, end: dt.datetime) -> str:
    total = max((end - start).total_seconds(), 1.0)
    pos = (ts - start).total_seconds() / total
    if pos < 1 / 3:
        return "fold_1"
    if pos < 2 / 3:
        return "fold_2"
    return "fold_3"


def rolling_atr_pcts(bars: Sequence[Bar], period: int) -> List[float]:
    out: List[float] = []
    trs: List[float] = []
    for i in range(1, len(bars)):
        curr = bars[i]
        prev = bars[i - 1]
        tr = max(curr.high - curr.low, abs(curr.high - prev.close), abs(curr.low - prev.close))
        trs.append(tr)
        if len(trs) >= period:
            a = max(mean(trs[-period:]), POINT * 5)
            out.append(a / max(curr.close, 1e-9))
    return out


def backtest(bars: Sequence[Bar], params: Params, label: str = "all") -> Tuple[List[Trade], Dict[str, Any], List[Snapshot]]:
    atr_pcts = rolling_atr_pcts(bars, params.atr_period)
    vol_thresholds = (percentile(atr_pcts, 0.33), percentile(atr_pcts, 0.67))

    equity = INITIAL_EQUITY
    peak = equity
    max_dd = 0.0
    position: Optional[Dict[str, Any]] = None
    cooldown_until = -1
    cooldown_bars = max(1, int(round(params.cooldown_minutes / 5)))
    trades: List[Trade] = []
    snapshots: List[Snapshot] = []
    start = bars[0].time
    end = bars[-1].time
    warmup = max(220, params.slow_sma + params.high_vol_breakout_lookback + params.atr_period + 5)

    for i in range(warmup, len(bars) - 1):
        hist = bars[max(0, i - 260) : i + 1]
        snapshot = build_snapshot(hist, params, vol_thresholds)
        if snapshot is None:
            continue
        snapshots.append(snapshot)
        nxt = bars[i + 1]

        if position is not None:
            side = str(position["side"])
            sl = float(position["sl"])
            tp = float(position["tp"])
            entry = float(position["entry"])
            volume = float(position["volume"])
            exit_price: Optional[float] = None
            reason = ""
            # Conservative intrabar ordering: SL before TP if both touched.
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
            if exit_price is None and bars_held >= 48:
                exit_price, reason = nxt.close, "TIME"

            if exit_price is not None:
                gross, net, commission, spread_cost = costs(entry, exit_price, volume, side)
                equity += net
                peak = max(peak, equity)
                max_dd = min(max_dd, equity / max(peak, 1e-9) - 1.0)
                risk_amount = abs(entry - float(position["orig_sl"])) * volume * CONTRACT_SIZE
                r_multiple = gross / max(risk_amount, 1e-9)
                trades.append(
                    Trade(
                        strategy=params.name,
                        fold=fold_for_time(dt.datetime.fromisoformat(str(position["entry_time"])), start, end),
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
                        r_multiple=r_multiple,
                        reason=reason,
                        bars_held=bars_held,
                        atr=float(position["atr"]),
                        atr_pct=float(position["atr_pct"]),
                        vol_bucket=str(position["vol_bucket"]),
                        session=str(position["session"]),
                    )
                )
                position = None
                cooldown_until = i + cooldown_bars

        if position is None and i >= cooldown_until and snapshot.signal in {"BUY", "SELL"}:
            entry = nxt.open
            sl_dist = snapshot.atr * params.stop_atr
            tp_dist = sl_dist * params.reward_multiple
            sl = entry - sl_dist if snapshot.signal == "BUY" else entry + sl_dist
            tp = entry + tp_dist if snapshot.signal == "BUY" else entry - tp_dist
            volume = position_size(equity, params, entry, sl)
            if volume > 0:
                position = {
                    "side": snapshot.signal,
                    "entry": entry,
                    "sl": sl,
                    "orig_sl": sl,
                    "tp": tp,
                    "volume": volume,
                    "entry_time": nxt.time.isoformat(),
                    "entry_i": i + 1,
                    "atr": snapshot.atr,
                    "atr_pct": snapshot.atr_pct,
                    "vol_bucket": snapshot.vol_bucket,
                    "session": snapshot.session,
                }

    metrics = summarize_trades(trades, equity, max_dd)
    metrics["label"] = label
    metrics["snapshots"] = len(snapshots)
    metrics["entry_signals"] = sum(1 for snap in snapshots if snap.signal != "NONE")
    metrics["raw_signals"] = sum(1 for snap in snapshots if snap.raw_signal != "NONE")
    metrics["high_vol_bars_pct"] = round(100.0 * sum(1 for snap in snapshots if snap.high_volatility) / max(len(snapshots), 1), 2)
    return trades, metrics, snapshots


def summarize_trades(trades: Sequence[Trade], final_equity: float, max_dd: float) -> Dict[str, Any]:
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
    out: Dict[str, Dict[str, Any]] = {}
    for group, rows in sorted(groups.items()):
        final_equity = INITIAL_EQUITY + sum(trade.net_pnl for trade in rows)
        out[group] = summarize_trades(rows, final_equity, 0.0)
    return out


def m5_to_h1_proxy(m5_bars: Sequence[Bar]) -> List[Bar]:
    buckets: Dict[dt.datetime, List[Bar]] = defaultdict(list)
    for bar in m5_bars:
        key = bar.time.replace(minute=0, second=0, microsecond=0)
        buckets[key].append(bar)
    out = []
    for key in sorted(buckets):
        chunk = buckets[key]
        if not chunk:
            continue
        out.append(
            Bar(
                time=key,
                open=chunk[0].open,
                high=max(b.high for b in chunk),
                low=min(b.low for b in chunk),
                close=chunk[-1].close,
                tick_volume=sum(b.tick_volume for b in chunk),
            )
        )
    return out


def regime_distribution_h1(bars: Sequence[Bar]) -> Dict[str, Any]:
    atr_pcts = rolling_atr_pcts(bars, 14)
    ranges: List[float] = []
    returns: List[float] = []
    sessions = Counter()
    months = Counter()
    for i in range(15, len(bars)):
        ranges.append((bars[i].high - bars[i].low) / max(bars[i].close, 1e-9))
        returns.append((bars[i].close / max(bars[i - 1].close, 1e-9)) - 1.0)
        sessions[session_of(bars[i].time)] += 1
        months[month_key(bars[i].time)] += 1
    return {
        "bars": len(bars),
        "start": bars[0].time.isoformat() if bars else None,
        "end": bars[-1].time.isoformat() if bars else None,
        "atr_pct_mean": round(mean(atr_pcts), 6),
        "atr_pct_p33": round(percentile(atr_pcts, 0.33), 6),
        "atr_pct_p67": round(percentile(atr_pcts, 0.67), 6),
        "atr_pct_p90": round(percentile(atr_pcts, 0.90), 6),
        "range_pct_mean": round(mean(ranges), 6),
        "abs_return_mean": round(mean([abs(r) for r in returns]), 6),
        "session_counts": dict(sessions),
        "month_count": len(months),
    }


def ks_distance(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    if not values_a or not values_b:
        return 0.0
    a = sorted(values_a)
    b = sorted(values_b)
    points = sorted(set(a + b))
    ia = 0
    ib = 0
    max_diff = 0.0
    for point in points:
        while ia < len(a) and a[ia] <= point:
            ia += 1
        while ib < len(b) and b[ib] <= point:
            ib += 1
        max_diff = max(max_diff, abs(ia / len(a) - ib / len(b)))
    return round(max_diff, 4)


def representativeness(m5_bars: Sequence[Bar], h1_bars: Sequence[Bar]) -> Dict[str, Any]:
    sample_h1 = m5_to_h1_proxy(m5_bars)
    pop = regime_distribution_h1(h1_bars)
    sample = regime_distribution_h1(sample_h1)
    pop_atr = rolling_atr_pcts(h1_bars, 14)
    sample_atr = rolling_atr_pcts(sample_h1, 14)
    sample_months = sorted({month_key(bar.time) for bar in m5_bars})
    return {
        "m5_sample": {
            "bars": len(m5_bars),
            "start": m5_bars[0].time.isoformat(),
            "end": m5_bars[-1].time.isoformat(),
            "months": len(sample_months),
            "month_first": sample_months[0],
            "month_last": sample_months[-1],
        },
        "sample_h1_proxy": sample,
        "long_h1_population_proxy": pop,
        "atr_pct_ks_distance_sample_vs_long_h1": ks_distance(sample_atr, pop_atr),
        "verdict": "sample_not_population" if len(sample_months) < 18 or ks_distance(sample_atr, pop_atr) > 0.18 else "sample_reasonably_representative",
    }


def wilson_interval(wins: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    phat = wins / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return round((centre - margin) * 100.0, 2), round((centre + margin) * 100.0, 2)


def add_inference_flags(metrics: Dict[str, Any], trades: Sequence[Trade]) -> Dict[str, Any]:
    wins = sum(1 for trade in trades if trade.net_pnl > 0)
    lo, hi = wilson_interval(wins, len(trades))
    out = dict(metrics)
    out["win_rate_95ci_low"] = lo
    out["win_rate_95ci_high"] = hi
    out["sample_flag"] = "LOW_N" if len(trades) < 60 else "OK_N"
    out["dd_guard_breached"] = abs(float(metrics.get("max_dd_pct", 0.0))) >= 5.0
    out["daily_guard_reference"] = "not fully simulated; inspect day-level pnl CSV if deploying"
    return out


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
        Params(name="active_sprint_v4"),
        Params(name="stricter_hv_gate", high_vol_atr_pct=0.0018, high_vol_min_breakout_atr=0.25, high_vol_min_close_location=0.68, high_vol_spike_atr=2.8),
        Params(name="looser_hv_gate", high_vol_atr_pct=0.0012, high_vol_min_momentum=0.55, high_vol_min_breakout_atr=0.10, high_vol_min_close_location=0.55),
        Params(name="active_rr_1p6", reward_multiple=1.6),
        Params(name="active_rr_2p8", reward_multiple=2.8),
        Params(name="no_high_vol_filter", high_vol_only=False),
    ]

    summary_rows: List[Dict[str, Any]] = []
    all_trades: List[Trade] = []
    diagnostics: Dict[str, Any] = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "symbol": SYMBOL,
        "representativeness": representativeness(m5_bars, h1_bars),
        "strategy_grouped": {},
    }

    for params in variants:
        trades, metrics, snapshots = backtest(m5_bars, params)
        metrics = add_inference_flags(metrics, trades)
        summary_rows.append({"strategy": params.name, **metrics})
        all_trades.extend(trades)
        diagnostics["strategy_grouped"][params.name] = {
            "fold": grouped_metrics(trades, "fold"),
            "session": grouped_metrics(trades, "session"),
            "vol_bucket": grouped_metrics(trades, "vol_bucket"),
            "months_with_trades": len({trade.entry_time[:7] for trade in trades}),
            "trade_month_counts": dict(Counter(trade.entry_time[:7] for trade in trades)),
        }

    summary_fields = list(summary_rows[0].keys()) if summary_rows else []
    write_csv(OUT_DIR / "doomsday_v4_sampling_comparison_summary.csv", summary_rows, summary_fields)
    trade_fields = [field.name for field in dataclasses.fields(Trade)]
    write_csv(OUT_DIR / "doomsday_v4_sampling_comparison_trades.csv", [dataclasses.asdict(t) for t in all_trades], trade_fields)
    with (OUT_DIR / "doomsday_v4_sampling_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    print(json.dumps({"summary": summary_rows, "representativeness": diagnostics["representativeness"], "out_dir": str(OUT_DIR)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
