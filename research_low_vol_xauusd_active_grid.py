#!/usr/bin/env python3
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path("/home/chain4655/Documents/Projects/MT5")
DATA_PATH = Path("/home/chain4655/Documents/backtest_reports/twitter_xauusd_candidates/20260520_225323/input_xauusd_m5.csv")
OUT_ROOT = Path("/home/chain4655/Documents/backtest_reports/xauusd_low_vol_active_grid")
INITIAL_EQUITY = 10000.0
CONTRACT_SIZE = 100.0
POINT = 0.01
ROUNDTRIP_COMMISSION_PER_LOT = 6.0
MAX_LOTS = 1.2


@dataclasses.dataclass(frozen=True)
class Bar:
    time: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    week: str


@dataclasses.dataclass(frozen=True)
class Trade:
    strategy: str
    param_id: str
    entry_time: dt.datetime
    exit_time: dt.datetime
    week: str
    side: str
    entry: float
    exit: float
    sl: float
    tp: float
    lots: float
    gross_pnl: float
    commission: float
    net_pnl: float
    r_multiple: float
    reason: str
    bars_held: int


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sma(values: Sequence[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    return statistics.fmean(values[-period:])


def ema(values: Sequence[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    k = 2.0 / (period + 1.0)
    out = statistics.fmean(values[:period])
    for value in values[period:]:
        out = value * k + out * (1.0 - k)
    return out


def atr(bars: Sequence[Bar], period: int) -> Optional[float]:
    if len(bars) < period + 1:
        return None
    vals: List[float] = []
    for i in range(-period, 0):
        curr = bars[i]
        prev = bars[i - 1]
        vals.append(max(curr.high - curr.low, abs(curr.high - prev.close), abs(curr.low - prev.close)))
    return max(statistics.fmean(vals), POINT * 5)


def week_key(ts: dt.datetime) -> str:
    year, week, _ = ts.isocalendar()
    return f"{year}-W{week:02d}"


def load_bars(path: Path = DATA_PATH) -> List[Bar]:
    data = pd.read_csv(path)
    data.columns = [str(c).strip().lower() for c in data.columns]
    if "time" not in data.columns:
        raise RuntimeError(f"No time column in {path}")
    data["time"] = pd.to_datetime(data["time"])
    data = data.sort_values("time")
    bars: List[Bar] = []
    for row in data.itertuples(index=False):
        t = getattr(row, "time").to_pydatetime()
        volume = float(getattr(row, "volume", 0.0) or 0.0)
        bars.append(
            Bar(
                time=t,
                open=float(getattr(row, "open")),
                high=float(getattr(row, "high")),
                low=float(getattr(row, "low")),
                close=float(getattr(row, "close")),
                volume=volume,
                week=week_key(t),
            )
        )
    return bars


def weekly_volatility_rows(bars: Sequence[Bar], threshold_pct: float = 10.0) -> Tuple[List[Dict[str, object]], set[str]]:
    grouped: Dict[str, List[Bar]] = {}
    for bar in bars:
        grouped.setdefault(bar.week, []).append(bar)
    rows: List[Dict[str, object]] = []
    low_weeks: set[str] = set()
    for wk, group in sorted(grouped.items()):
        if len(group) < 200:  # partial holiday/week fragment; keep row but mark sample weak
            sample_ok = False
        else:
            sample_ok = True
        open_px = group[0].open
        high = max(b.high for b in group)
        low = min(b.low for b in group)
        close = group[-1].close
        range_pct = (high - low) / max(open_px, 1e-9) * 100.0
        close_to_close_pct = abs(close / max(open_px, 1e-9) - 1.0) * 100.0
        low_vol = range_pct < threshold_pct and sample_ok
        if low_vol:
            low_weeks.add(wk)
        rows.append(
            {
                "week": wk,
                "start": group[0].time.isoformat(sep=" "),
                "end": group[-1].time.isoformat(sep=" "),
                "bars": len(group),
                "open": round(open_px, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "range_pct": round(range_pct, 3),
                "abs_close_to_close_pct": round(close_to_close_pct, 3),
                "low_vol_lt_10pct": low_vol,
                "sample_ok": sample_ok,
            }
        )
    return rows, low_weeks


class Strategy:
    name = "base"
    risk_pct = 0.005
    stop_atr = 2.5
    reward_multiple = 2.5
    trail_trigger_atr = 1.5
    trail_lock_atr = 0.5
    break_even_atr = 999.0
    break_even_lock_atr = 0.0
    max_hold_bars = 36
    cooldown_bars = 2

    def __init__(self, params: Dict[str, object]):
        self.params = params
        for k, v in params.items():
            setattr(self, k, v)
        self.param_id = self.build_param_id()

    def build_param_id(self) -> str:
        body = ";".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}|{body}"

    def signal(self, hist: Sequence[Bar]) -> Tuple[str, float, float]:
        return "NONE", 0.0, 0.0

    def sl_tp(self, side: str, entry: float, a: float, _score: float) -> Tuple[float, float]:
        stop = a * float(self.stop_atr)
        target = stop * float(self.reward_multiple)
        if side == "BUY":
            return entry - stop, entry + target
        return entry + stop, entry - target


class TrendStrategy(Strategy):
    name = "trend_following_active"
    risk_pct = 0.0075
    fast_sma = 20
    slow_sma = 60
    htf_fast_sma = 50
    htf_slow_sma = 200
    threshold = 0.35
    atr_period = 14
    breakout_lookback = 20
    stop_atr = 2.5
    reward_multiple = 2.5
    trail_trigger_atr = 1.5
    trail_lock_atr = 0.5
    break_even_atr = 1.0
    break_even_lock_atr = 0.15
    max_hold_bars = 36
    cooldown_bars = 2

    def _htf_signal(self, hist: Sequence[Bar]) -> str:
        # M5 → H1 proxy: take every 12th close. This avoids live MT5 H1 calls while preserving direction filter.
        closes = [b.close for b in hist]
        hourly = closes[::12]
        if len(hourly) < max(int(self.htf_fast_sma), int(self.htf_slow_sma)):
            return "NEUTRAL"
        fast = sma(hourly, int(self.htf_fast_sma)) or hourly[-1]
        slow = sma(hourly, int(self.htf_slow_sma)) or hourly[-1]
        if fast > slow:
            return "BULL"
        if fast < slow:
            return "BEAR"
        return "NEUTRAL"

    def signal(self, hist: Sequence[Bar]) -> Tuple[str, float, float]:
        if len(hist) < max(int(self.slow_sma), int(self.breakout_lookback), int(self.atr_period)) + 5:
            return "NONE", 0.0, 0.0
        closes = [b.close for b in hist]
        highs = [b.high for b in hist]
        lows = [b.low for b in hist]
        a = atr(hist, int(self.atr_period))
        if a is None:
            return "NONE", 0.0, 0.0
        fast = sma(closes, int(self.fast_sma)) or closes[-1]
        slow = sma(closes, int(self.slow_sma)) or closes[-1]
        momentum = clamp((closes[-1] - closes[-4]) / a, -2.0, 2.0)
        htf = self._htf_signal(hist)
        trend = 0.55 if closes[-1] > fast > slow else (-0.55 if closes[-1] < fast < slow else (0.20 if closes[-1] > slow else -0.20))
        htf_bias = 0.35 if htf == "BULL" else (-0.35 if htf == "BEAR" else 0.0)
        lookback = max(10, int(self.breakout_lookback))
        recent_high = max(highs[-lookback:])
        recent_low = min(lows[-lookback:])
        breakout = clamp((closes[-1] - recent_high) / a, -1.0, 1.0) * 0.30
        breakout += clamp((recent_low - closes[-1]) / a, -1.0, 1.0) * -0.30
        score = clamp(trend + htf_bias + momentum * 0.25 + breakout, -1.5, 1.5)
        if htf == "BULL" and score >= float(self.threshold):
            return "BUY", score, a
        if htf == "BEAR" and score <= -float(self.threshold):
            return "SELL", score, a
        return "NONE", score, a


class DoomsdayStrategy(Strategy):
    name = "doomsday_active"
    risk_pct = 0.005
    fast_sma = 7
    slow_sma = 30
    atr_period = 14
    threshold = 0.60
    stop_atr = 3.0
    reward_multiple = 2.4
    long_bias = 0.68
    high_vol_atr_pct = 0.0021
    high_vol_range_atr = 4.75
    breakout_lookback = 16
    min_momentum = 0.85
    spike_atr = 3.0
    min_breakout_atr = 0.35
    min_close_location = 0.68
    max_hold_bars = 30
    cooldown_bars = 5

    def signal(self, hist: Sequence[Bar]) -> Tuple[str, float, float]:
        if len(hist) < max(int(self.slow_sma), int(self.breakout_lookback), int(self.atr_period)) + 5:
            return "NONE", 0.0, 0.0
        closes = [b.close for b in hist]
        highs = [b.high for b in hist]
        lows = [b.low for b in hist]
        a = atr(hist, int(self.atr_period))
        if a is None or a <= 0:
            return "NONE", 0.0, 0.0
        fast = sma(closes, int(self.fast_sma)) or closes[-1]
        slow = sma(closes, int(self.slow_sma)) or closes[-1]
        momentum = clamp((closes[-1] - closes[-4]) / a, -2.0, 2.0)
        range_lookback = min(20, len(hist))
        range_atr = (max(highs[-range_lookback:]) - min(lows[-range_lookback:])) / a
        last = hist[-1]
        last_range = max(last.high - last.low, 1e-9)
        spike = last_range / a
        high_vol = (a / max(last.close, 1e-9) >= float(self.high_vol_atr_pct)) and (range_atr >= float(self.high_vol_range_atr) or spike >= float(self.spike_atr))
        if not high_vol or abs(momentum) < float(self.min_momentum):
            return "NONE", 0.0, a
        lookback = max(5, int(self.breakout_lookback))
        recent_high = max(highs[-lookback - 1 : -1])
        recent_low = min(lows[-lookback - 1 : -1])
        trend = 0.40 if closes[-1] > fast > slow else (-0.40 if closes[-1] < fast < slow else (0.12 if closes[-1] > slow else -0.12))
        up_break = clamp((closes[-1] - recent_high) / a, -1.5, 1.5)
        dn_break = clamp((recent_low - closes[-1]) / a, -1.5, 1.5)
        breakout = up_break * 0.45 + dn_break * -0.45
        score = clamp(trend + breakout + momentum * 0.45 + (float(self.long_bias) - 0.5) * 0.20, -1.5, 1.5)
        close_loc = (last.close - last.low) / last_range
        if score >= float(self.threshold) and momentum > 0 and up_break >= float(self.min_breakout_atr) and close_loc >= float(self.min_close_location):
            return "BUY", score, a
        if score <= -float(self.threshold) and momentum < 0 and dn_break <= -float(self.min_breakout_atr) and close_loc <= (1.0 - float(self.min_close_location)):
            return "SELL", score, a
        return "NONE", score, a


class MomentumSurferStrategy(Strategy):
    name = "momentum_surfer_active"
    risk_pct = 0.0025
    atr_period = 14
    vol_lookback = 20
    mom_lookback = 12
    accel_min = 0.15
    stop_atr = 2.0
    reward_multiple = 2.5
    trail_trigger_atr = 1.5
    trail_lock_atr = 0.5
    max_hold_bars = 36
    cooldown_bars = 1

    def signal(self, hist: Sequence[Bar]) -> Tuple[str, float, float]:
        need = max(int(self.atr_period), int(self.vol_lookback), int(self.mom_lookback)) + 10
        if len(hist) < need:
            return "NONE", 0.0, 0.0
        closes = [b.close for b in hist]
        a = atr(hist, int(self.atr_period))
        base_slice = hist[: -max(1, int(self.mom_lookback))]
        baseline = atr(base_slice, min(int(self.vol_lookback), int(self.atr_period))) if len(base_slice) > int(self.atr_period) + 2 else a
        if a is None or baseline is None or a <= 0:
            return "NONE", 0.0, 0.0
        vol_ratio = a / max(baseline, 1e-9)
        if vol_ratio < 1.0:
            return "NONE", 0.0, a
        mom_1 = closes[-1] - closes[-2]
        mom_2 = closes[-2] - closes[-3]
        mom_3 = closes[-3] - closes[-4]
        accel = mom_1 - mom_2
        accel_norm = abs(accel) / a
        align = 0.3 if (mom_1 > 0 and mom_2 > 0 and mom_3 > 0) or (mom_1 < 0 and mom_2 < 0 and mom_3 < 0) else 0.0
        strength = accel_norm + align + (abs(mom_1) / a) * 0.3 + max(0.0, vol_ratio - 1.0) * 0.4
        if strength < float(self.accel_min):
            return "NONE", strength, a
        if mom_1 > 0 and mom_2 > 0 and accel > 0:
            return "BUY", strength, a
        if mom_1 < 0 and mom_2 < 0 and accel < 0:
            return "SELL", -strength, a
        return "NONE", strength, a


def position_size(equity: float, risk_pct: float, entry: float, sl: float) -> float:
    risk_per_lot = abs(entry - sl) * CONTRACT_SIZE
    if risk_per_lot <= 0:
        return 0.0
    raw = equity * risk_pct / risk_per_lot
    return max(0.01, min(MAX_LOTS, math.floor(raw / 0.01) * 0.01))


def backtest_strategy(bars: Sequence[Bar], low_weeks: set[str], strat: Strategy) -> Tuple[List[Trade], Dict[str, object]]:
    equity = INITIAL_EQUITY
    peak = equity
    max_dd = 0.0
    trades: List[Trade] = []
    pos: Optional[Dict[str, object]] = None
    cooldown_until = -1
    warmup = 2600  # enough for H1 200-SMA proxy on M5
    for i in range(min(warmup, max(0, len(bars) - 2)), len(bars) - 1):
        # Keep a bounded rolling window; all signals need at most ~2600 M5 bars
        # (H1 200-SMA proxy = 2400 M5 bars). Avoid O(n^2) full-history slicing.
        hist = bars[max(0, i - 3000) : i + 1]
        bar = bars[i]
        nxt = bars[i + 1]
        if pos is not None:
            side = str(pos["side"])
            sl = float(pos["sl"])
            tp = float(pos["tp"])
            entry = float(pos["entry"])
            entry_i = int(pos["entry_i"])
            a_entry = float(pos["atr"])
            exit_price: Optional[float] = None
            reason = ""
            if side == "BUY":
                profit_move = bar.close - entry
                if profit_move >= a_entry * float(strat.break_even_atr):
                    sl = max(sl, entry + a_entry * float(strat.break_even_lock_atr))
                if profit_move >= a_entry * float(strat.trail_trigger_atr):
                    sl = max(sl, entry + a_entry * float(strat.trail_lock_atr))
                pos["sl"] = sl
                if nxt.low <= sl:
                    exit_price, reason = sl, "SL_TRAIL"
                elif nxt.high >= tp:
                    exit_price, reason = tp, "TP"
            else:
                profit_move = entry - bar.close
                if profit_move >= a_entry * float(strat.break_even_atr):
                    sl = min(sl, entry - a_entry * float(strat.break_even_lock_atr))
                if profit_move >= a_entry * float(strat.trail_trigger_atr):
                    sl = min(sl, entry - a_entry * float(strat.trail_lock_atr))
                pos["sl"] = sl
                if nxt.high >= sl:
                    exit_price, reason = sl, "SL_TRAIL"
                elif nxt.low <= tp:
                    exit_price, reason = tp, "TP"
            bars_held = i - entry_i
            if exit_price is None and bars_held >= int(strat.max_hold_bars):
                exit_price, reason = nxt.close, "TIME"
            if exit_price is not None:
                lots = float(pos["lots"])
                mult = 1.0 if side == "BUY" else -1.0
                gross = (exit_price - entry) * mult * lots * CONTRACT_SIZE
                commission = ROUNDTRIP_COMMISSION_PER_LOT * lots
                net = gross - commission
                risk_amount = abs(entry - float(pos["initial_sl"])) * lots * CONTRACT_SIZE
                r_mult = net / max(risk_amount, 1e-9)
                equity += net
                peak = max(peak, equity)
                max_dd = min(max_dd, equity / peak - 1.0)
                trades.append(
                    Trade(
                        strategy=strat.name,
                        param_id=strat.param_id,
                        entry_time=dt.datetime.fromisoformat(str(pos["entry_time"])),
                        exit_time=nxt.time,
                        week=str(pos["week"]),
                        side=side,
                        entry=entry,
                        exit=float(exit_price),
                        sl=sl,
                        tp=tp,
                        lots=lots,
                        gross_pnl=gross,
                        commission=commission,
                        net_pnl=net,
                        r_multiple=r_mult,
                        reason=reason,
                        bars_held=bars_held,
                    )
                )
                pos = None
                cooldown_until = i + int(strat.cooldown_bars)
        if pos is None and i >= cooldown_until and nxt.week in low_weeks:
            sig, score, a = strat.signal(hist)
            if sig in {"BUY", "SELL"} and a > 0:
                entry = nxt.open
                sl, tp = strat.sl_tp(sig, entry, a, score)
                lots = position_size(equity, float(strat.risk_pct), entry, sl)
                if lots > 0:
                    pos = {
                        "side": sig,
                        "entry": entry,
                        "sl": sl,
                        "initial_sl": sl,
                        "tp": tp,
                        "lots": lots,
                        "entry_time": nxt.time.isoformat(),
                        "entry_i": i + 1,
                        "week": nxt.week,
                        "atr": a,
                    }
    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]
    gross_profit = sum(t.net_pnl for t in wins)
    gross_loss = -sum(t.net_pnl for t in losses)
    metrics = {
        "strategy": strat.name,
        "param_id": strat.param_id,
        "trades": len(trades),
        "net_pnl": round(equity - INITIAL_EQUITY, 2),
        "return_pct": round((equity / INITIAL_EQUITY - 1.0) * 100.0, 3),
        "win_rate_pct": round(len(wins) / len(trades) * 100.0, 2) if trades else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0),
        "max_dd_pct": round(max_dd * 100.0, 3),
        "avg_r": round(statistics.fmean([t.r_multiple for t in trades]), 4) if trades else 0.0,
        "expectancy_usd": round(statistics.fmean([t.net_pnl for t in trades]), 2) if trades else 0.0,
    }
    return trades, metrics


def strategy_grid() -> Iterable[Strategy]:
    # Active-plan centered grids. This is exhaustive over the explicit research grid below, not infinite CLI space.
    for fast, slow, th, stop, rr, lookback in itertools.product(
        [20, 28], [60, 80], [0.30, 0.35, 0.45], [2.0, 2.5, 3.0], [2.0, 2.5, 3.0], [20]
    ):
        if fast < slow:
            yield TrendStrategy({"fast_sma": fast, "slow_sma": slow, "threshold": th, "stop_atr": stop, "reward_multiple": rr, "breakout_lookback": lookback})
    for th, stop, rr, hv_atr, min_mom in itertools.product(
        [0.55, 0.60], [2.5, 3.0], [2.0, 2.4], [0.0016, 0.0021], [0.65, 0.85]
    ):
        yield DoomsdayStrategy({"threshold": th, "stop_atr": stop, "reward_multiple": rr, "high_vol_atr_pct": hv_atr, "min_momentum": min_mom})
    for accel_min, stop, rr, mom, trail in itertools.product(
        [0.12, 0.15, 0.20], [1.5, 2.0, 2.5], [2.0, 2.5, 3.0], [8], [0.4, 0.5]
    ):
        yield MomentumSurferStrategy({"accel_min": accel_min, "stop_atr": stop, "reward_multiple": rr, "mom_lookback": mom, "trail_lock_atr": trail})


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    bars = load_bars()
    weekly_rows, low_weeks = weekly_volatility_rows(bars, threshold_pct=10.0)
    if not low_weeks:
        raise RuntimeError("No weekly volatility <10% weeks found in dataset")
    write_csv(out_dir / "weekly_volatility.csv", weekly_rows)

    summary_rows: List[Dict[str, object]] = []
    top_trades: List[Trade] = []
    total = 0
    for strat in strategy_grid():
        total += 1
        trades, metrics = backtest_strategy(bars, low_weeks, strat)
        summary_rows.append(metrics)
        if trades and (len(top_trades) < 1 or float(metrics["return_pct"]) >= 0):
            top_trades.extend(trades)

    summary_rows = sorted(
        summary_rows,
        key=lambda r: (float(r["return_pct"]), float(r["profit_factor"]), int(r["trades"]), float(r["max_dd_pct"])),
        reverse=True,
    )
    write_csv(out_dir / "grid_summary.csv", summary_rows)
    write_csv(out_dir / "top100_summary.csv", summary_rows[:100])

    # Save trades only for top 10 parameter sets to keep file readable.
    top_param_ids = {str(r["param_id"]) for r in summary_rows[:10]}
    selected_trades: List[Dict[str, object]] = []
    for strat in strategy_grid():
        if strat.param_id not in top_param_ids:
            continue
        trades, _metrics = backtest_strategy(bars, low_weeks, strat)
        for t in trades:
            row = dataclasses.asdict(t)
            row["entry_time"] = t.entry_time.isoformat(sep=" ")
            row["exit_time"] = t.exit_time.isoformat(sep=" ")
            row["gross_pnl"] = round(t.gross_pnl, 2)
            row["commission"] = round(t.commission, 2)
            row["net_pnl"] = round(t.net_pnl, 2)
            row["r_multiple"] = round(t.r_multiple, 4)
            selected_trades.append(row)
    write_csv(out_dir / "top10_trades.csv", selected_trades)

    low_week_rows = [r for r in weekly_rows if r["low_vol_lt_10pct"]]
    report_lines = [
        "# XAUUSD low-volatility (<10% weekly range) active MT5 grid backtest",
        "",
        f"Data: `{DATA_PATH}`",
        f"Bars: {len(bars)} | range: {bars[0].time} → {bars[-1].time}",
        f"Low-vol weeks: {len(low_weeks)} / {len(weekly_rows)} weeks (weekly high-low / week open < 10%, partial weeks excluded)",
        f"Parameter sets tested: {total}",
        "Cost model: XAUUSD contract_size=100, roundtrip commission=$6/lot; spread/slippage not separately modeled due no bid/ask series.",
        "",
        "## Top 10 parameter sets",
    ]
    for idx, row in enumerate(summary_rows[:10], start=1):
        report_lines.append(
            f"{idx}. {row['strategy']}: return={row['return_pct']}%, PF={row['profit_factor']}, trades={row['trades']}, "
            f"win={row['win_rate_pct']}%, maxDD={row['max_dd_pct']}%, avgR={row['avg_r']} | {row['param_id']}"
        )
    report_lines.extend([
        "",
        "## Low-volatility weeks",
    ])
    for row in low_week_rows:
        report_lines.append(
            f"- {row['week']}: {row['start']} → {row['end']} | range={row['range_pct']}% | close_move={row['abs_close_to_close_pct']}% | bars={row['bars']}"
        )
    report_lines.extend([
        "",
        "## Files",
        "- `weekly_volatility.csv`: all weekly volatility buckets.",
        "- `grid_summary.csv`: all parameter-set results.",
        "- `top100_summary.csv`: ranked top 100.",
        "- `top10_trades.csv`: trades for top 10 parameter sets.",
    ])
    (out_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "bars": len(bars),
                "start": bars[0].time.isoformat(),
                "end": bars[-1].time.isoformat(),
                "low_vol_week_count": len(low_weeks),
                "parameter_sets_tested": total,
                "top10": summary_rows[:10],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(out_dir), "top10": summary_rows[:10], "low_vol_weeks": len(low_weeks), "parameter_sets": total}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
