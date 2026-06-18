#!/usr/bin/env python3.14
"""Asian Session pivot-stall reversal BACKTEST — rewrite v2.

v1 tried session-level extremes and found 0 trades (too strict).
Rewrite: detect LOCAL rolling pivot highs/lows, check for post-pivot
stall (tight candles, declining volume), then fade the reversal.

Two sub-strategies:
  A) Pivot Stall — after a local high/low, N tight bars, then fade
  B) V-Reversal  — sharp 1-2 bar spike to extreme, immediate bounce

Read-only research; no MT5 orders.
"""
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import math
import statistics
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from pymt5linux import MetaTrader5
except ImportError:
    sys.path.insert(0, "/home/chain4655/.openharness-venv/lib64/python3.14/site-packages")
    from pymt5linux import MetaTrader5

ROOT = Path("/home/chain4655/Documents/Projects/MT5")
OUT_DIR = ROOT / "backtest_reports_asian_reversal_v2"
SYMBOL = "XAUUSD"
INITIAL_EQUITY = 100000.0
CONTRACT_SIZE = 100.0
POINT = 0.01
SPREAD_POINTS = 20.0
COMMISSION_PER_LOT = 7.0
MAX_M1_BARS = 80000


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
    mode: str = "pivot_stall"   # "pivot_stall" or "v_reversal"

    # Pivot detection
    pivot_lookback: int = 4
    stall_bars: int = 3
    stall_range_atr: float = 0.40
    stall_vol_ratio: float = 0.80
    reversal_atr: float = 0.25          # min reversal bar move as fraction of ATR

    # V-reversal
    v_spike_range_atr: float = 0.80     # spike bar range must exceed this * ATR
    v_recovery_range_atr: float = 0.25  # next bar range below this to confirm recovery
    v_min_bounce_atr: float = 0.35      # price must bounce at least this * ATR from low

    # Risk
    stop_atr: float = 2.0
    take_profit_atr: float = 4.0
    max_hold_bars: int = 48
    risk_pct: float = 0.0035

    # Session
    session_start_hour: int = 0
    session_end_hour: int = 7
    max_trades_per_day: int = 1

    # Filter
    min_atr: float = 0.50         # skip if ATR too small (dead market)
    max_spread_atr: float = 0.15  # skip if spread too wide relative to ATR


@dataclasses.dataclass(frozen=True)
class Trade:
    params: str
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
    pivot_level: float
    pivot_type: str  # "high_stall", "low_stall", "v_sell", "v_buy"
    stall_avg_range_atr: float
    stall_vol_ratio: float


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def true_ranges(bars: Sequence[Bar]) -> List[float]:
    out: List[float] = []
    for i in range(1, len(bars)):
        curr, prev = bars[i], bars[i - 1]
        out.append(max(curr.high - curr.low, abs(curr.high - prev.close), abs(curr.low - prev.close)))
    return out


def atr_from_bars(bars: Sequence[Bar], period: int) -> Optional[float]:
    if len(bars) < period + 1:
        return None
    trs = true_ranges(bars[-(period + 1):])
    return max(mean(trs), POINT * 5)


def fetch_bars(timeframe: str, count: int) -> List[Bar]:
    mt5 = MetaTrader5(host="127.0.0.1", port=18812)
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        tf_map = {"M1": mt5.TIMEFRAME_M1}
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


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    it = iter(rows)
    try:
        first = next(it)
    except StopIteration:
        path.write_text("", encoding="utf-8")
        return
    fields = list(first.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields, extrasaction="ignore")
        w.writeheader()
        w.writerow(first)
        w.writerows(it)


def is_pivot_high(bars: List[Bar], i: int, lookback: int) -> bool:
    if i < lookback or i >= len(bars) - lookback:
        return False
    for offset in range(1, lookback + 1):
        if bars[i].high <= bars[i - offset].high or bars[i].high <= bars[i + offset].high:
            return False
    return True


def is_pivot_low(bars: List[Bar], i: int, lookback: int) -> bool:
    if i < lookback or i >= len(bars) - lookback:
        return False
    for offset in range(1, lookback + 1):
        if bars[i].low >= bars[i - offset].low or bars[i].low >= bars[i + offset].low:
            return False
    return True


def detect_v_reversal(
    bars: List[Bar], i: int, atr: float, params: Params,
) -> Optional[Tuple[str, float, float]]:
    """Check if bar[i] completes a V-reversal pattern.
    Returns (side, entry_price, pivot_price) or None.
    """
    if i < 3 or i >= len(bars) - 1:
        return None
    spike = bars[i - 1]
    curr = bars[i]
    prev = bars[i - 2] if i >= 2 else None
    if prev is None:
        return None

    spike_range = spike.high - spike.low
    if spike_range < params.v_spike_range_atr * atr:
        return None

    # Check for V-bottom (BUY): spike bar made a new low, curr closes higher
    low_buy = False
    high_sell = False

    if spike.low < prev.low and spike.low < curr.low:
        bounce = curr.close - spike.low
        if curr.close > curr.open and bounce >= params.v_min_bounce_atr * atr:
            curr_range = curr.high - curr.low
            if curr_range <= params.v_recovery_range_atr * atr:
                low_buy = True

    # Check for V-top (SELL): spike bar made a new high, curr closes lower
    if spike.high > prev.high and spike.high > curr.high:
        bounce = spike.high - curr.close
        if curr.close < curr.open and bounce >= params.v_min_bounce_atr * atr:
            curr_range = curr.high - curr.low
            if curr_range <= params.v_recovery_range_atr * atr:
                high_sell = True

    if low_buy:
        return ("BUY", curr.close, spike.low)
    if high_sell:
        return ("SELL", curr.close, spike.high)
    return None


def detect_pivot_stall(
    bars: List[Bar], i: int, atr: float, params: Params,
    last_high_pivot: int, last_low_pivot: int,
) -> Optional[Tuple[str, float, float, float, float]]:
    """Check if bar[i] completes a pivot-stall-reversal pattern.
    Returns (side, entry_price, pivot_level, avg_stall_range_atr, stall_vol_ratio) or None.
    """
    # Check for SELL after pivot high
    if last_high_pivot >= 0 and i > last_high_pivot + params.stall_bars:
        stall_bars_slice = bars[last_high_pivot + 1:i + 1]
        if len(stall_bars_slice) < params.stall_bars:
            return None

        # All bars below pivot high (stall or decline)
        pivot_high = bars[last_high_pivot].high
        stall_bars_used = stall_bars_slice[-params.stall_bars:]

        stall_ranges = [b.high - b.low for b in stall_bars_used]
        avg_range_atr = mean(stall_ranges) / max(atr, POINT)
        if avg_range_atr > params.stall_range_atr:
            return None

        stall_vols = [b.tick_volume for b in stall_bars_used]
        # Compare stall vol to vol at pivot
        pivot_vol = bars[last_high_pivot].tick_volume
        vol_ratio = mean(stall_vols) / max(pivot_vol, 1)
        if vol_ratio > params.stall_vol_ratio:
            return None

        # Current bar must break down (close below pivot high - reversal_atr * atr)
        curr = bars[i]
        if curr.close < pivot_high - params.reversal_atr * atr and curr.close < curr.open:
            return ("SELL", curr.close, pivot_high, avg_range_atr, vol_ratio)

    # Check for BUY after pivot low
    if last_low_pivot >= 0 and i > last_low_pivot + params.stall_bars:
        stall_bars_slice = bars[last_low_pivot + 1:i + 1]
        if len(stall_bars_slice) < params.stall_bars:
            return None

        pivot_low = bars[last_low_pivot].low
        stall_bars_used = stall_bars_slice[-params.stall_bars:]

        stall_ranges = [b.high - b.low for b in stall_bars_used]
        avg_range_atr = mean(stall_ranges) / max(atr, POINT)
        if avg_range_atr > params.stall_range_atr:
            return None

        stall_vols = [b.tick_volume for b in stall_bars_used]
        pivot_vol = bars[last_low_pivot].tick_volume
        vol_ratio = mean(stall_vols) / max(pivot_vol, 1)
        if vol_ratio > params.stall_vol_ratio:
            return None

        curr = bars[i]
        if curr.close > pivot_low + params.reversal_atr * atr and curr.close > curr.open:
            return ("BUY", curr.close, pivot_low, avg_range_atr, vol_ratio)

    return None


def backtest(bars: List[Bar], params: Params) -> Tuple[List[Trade], Dict[str, Any]]:
    trades: List[Trade] = []
    last_trade_day: Optional[str] = None

    # Pivot tracking state
    last_high_pivot: int = -1
    last_low_pivot: int = -1

    min_start = max(params.pivot_lookback, params.stall_bars) + 5

    for i in range(min_start, len(bars)):
        curr = bars[i]
        if not (params.session_start_hour <= curr.time.hour < params.session_end_hour):
            continue

        day_key = curr.time.date().isoformat()
        if day_key == last_trade_day:
            # Still update pivots
            if is_pivot_high(bars, i, params.pivot_lookback):
                last_high_pivot = i
            if is_pivot_low(bars, i, params.pivot_lookback):
                last_low_pivot = i
            continue

        # New session or same session, check pivots
        if is_pivot_high(bars, i, params.pivot_lookback):
            last_high_pivot = i
        if is_pivot_low(bars, i, params.pivot_lookback):
            last_low_pivot = i

        atr = atr_from_bars(bars[:i], 14)
        if atr is None or atr < params.min_atr:
            continue

        entry: Optional[Tuple[str, float, float, float, float, str]] = None

        if params.mode == "pivot_stall":
            result = detect_pivot_stall(bars, i, atr, params, last_high_pivot, last_low_pivot)
            if result is not None:
                side, entry_price, pivot_level, stall_range, vol_ratio = result
                pivot_type = "high_stall" if side == "SELL" else "low_stall"
                entry = (side, entry_price, pivot_level, stall_range, vol_ratio, pivot_type)
        elif params.mode == "v_reversal":
            result = detect_v_reversal(bars, i, atr, params)
            if result is not None:
                side, entry_price, pivot_level = result
                pivot_type = "v_buy" if side == "BUY" else "v_sell"
                entry = (side, entry_price, pivot_level, 0.0, 0.0, pivot_type)

        if entry is None:
            continue

        side, entry_price, pivot_level, stall_range, vol_ratio, pivot_type = entry

        sl_price = entry_price - params.stop_atr * atr if side == "BUY" else entry_price + params.stop_atr * atr
        tp_price = entry_price + params.take_profit_atr * atr if side == "BUY" else entry_price - params.take_profit_atr * atr

        risk_amount = INITIAL_EQUITY * params.risk_pct
        risk_points = abs(entry_price - sl_price)
        vol = clamp(risk_amount / max(risk_points, POINT) / CONTRACT_SIZE, 0.01, 10.0)

        exit_price = entry_price
        exit_idx = i
        reason = "max_hold"
        max_exit = min(i + params.max_hold_bars, len(bars))
        for j in range(i + 1, max_exit):
            bar = bars[j]
            if side == "BUY":
                if bar.low <= sl_price:
                    exit_price, exit_idx, reason = sl_price, j, "stop_loss"
                    break
                if bar.high >= tp_price:
                    exit_price, exit_idx, reason = tp_price, j, "take_profit"
                    break
            else:
                if bar.high >= sl_price:
                    exit_price, exit_idx, reason = sl_price, j, "stop_loss"
                    break
                if bar.low <= tp_price:
                    exit_price, exit_idx, reason = tp_price, j, "take_profit"
                    break
            exit_price, exit_idx = bar.close, j

        bars_held = exit_idx - i
        gross_pnl = ((exit_price - entry_price) if side == "BUY" else (entry_price - exit_price)) * vol * CONTRACT_SIZE
        spread_cost = SPREAD_POINTS * POINT * vol * CONTRACT_SIZE
        commission = COMMISSION_PER_LOT * vol
        net_pnl = gross_pnl - spread_cost - commission
        r_multiple = gross_pnl / max(risk_amount, 0.01)

        trades.append(Trade(
            params=params.name,
            entry_time=curr.time.isoformat(),
            exit_time=bars[min(exit_idx, len(bars) - 1)].time.isoformat(),
            side=side, entry=entry_price, exit=exit_price,
            sl=sl_price, tp=tp_price, volume=round(vol, 2),
            gross_pnl=round(gross_pnl, 2), net_pnl=round(net_pnl, 2),
            commission=round(commission, 2), spread_cost=round(spread_cost, 2),
            r_multiple=round(r_multiple, 2), reason=reason, bars_held=bars_held,
            atr=round(atr, 2), pivot_level=round(pivot_level, 2),
            pivot_type=pivot_type,
            stall_avg_range_atr=round(stall_range, 3),
            stall_vol_ratio=round(vol_ratio, 2),
        ))
        last_trade_day = day_key

    if not trades:
        return [], {"total_trades": 0, "total_net_pnl": 0.0, "win_rate": 0.0, "profit_factor": 0.0}

    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]
    total_gross = sum(t.gross_pnl for t in trades)
    total_net = sum(t.net_pnl for t in trades)
    gross_wins = sum(t.gross_pnl for t in wins)
    gross_losses = abs(sum(t.gross_pnl for t in losses))
    pf = gross_wins / max(gross_losses, 0.01)

    metrics = {
        "total_trades": len(trades),
        "win_count": len(wins), "loss_count": len(losses),
        "win_rate": round(len(wins) / max(len(trades), 1) * 100, 1),
        "total_gross_pnl": round(total_gross, 2),
        "total_net_pnl": round(total_net, 2),
        "profit_factor": round(pf, 2),
        "avg_r_multiple": round(mean([t.r_multiple for t in trades]), 2),
        "avg_bars_held": round(mean([t.bars_held for t in trades]), 1),
        "buy_ratio": round(sum(1 for t in trades if t.side == "BUY") / max(len(trades), 1) * 100, 1),
    }
    return trades, metrics


def main() -> None:
    print(f"Fetching {MAX_M1_BARS} M1 bars...")
    m1_bars = fetch_bars("M1", MAX_M1_BARS)
    print(f"Got {len(m1_bars)} bars ({m1_bars[0].time.date()} to {m1_bars[-1].time.date()}).")

    variants = [
        # Pivot Stall variants
        Params(name="stall_4lb_3b_0.4r_0.8v", mode="pivot_stall"),
        Params(name="stall_4lb_3b_0.5r_0.9v", mode="pivot_stall",
               stall_range_atr=0.5, stall_vol_ratio=0.9),
        Params(name="stall_4lb_5b_0.4r_0.8v", mode="pivot_stall",
               stall_bars=5),
        Params(name="stall_3lb_3b_0.4r_0.8v", mode="pivot_stall",
               pivot_lookback=3),
        Params(name="stall_4lb_3b_0.4r_0.8v_tp3", mode="pivot_stall",
               take_profit_atr=3.0),
        Params(name="stall_4lb_3b_0.4r_0.8v_sl1.5", mode="pivot_stall",
               stop_atr=1.5),
        Params(name="stall_5lb_3b_0.35r_0.7v", mode="pivot_stall",
               pivot_lookback=5, stall_range_atr=0.35, stall_vol_ratio=0.7),
        # V-reversal variants
        Params(name="vrev_0.8s_0.25r_0.35b", mode="v_reversal"),
        Params(name="vrev_1.0s_0.3r_0.5b", mode="v_reversal",
               v_spike_range_atr=1.0, v_recovery_range_atr=0.3, v_min_bounce_atr=0.5),
        Params(name="vrev_0.6s_0.2r_0.3b", mode="v_reversal",
               v_spike_range_atr=0.6, v_recovery_range_atr=0.2, v_min_bounce_atr=0.3),
    ]

    all_trades: List[Trade] = []
    summary_rows: List[Dict[str, object]] = []

    for p in variants:
        trades, metrics = backtest(m1_bars, p)
        metrics["params"] = p.name
        summary_rows.append(metrics)
        all_trades.extend(trades)
        print(f"  {p.name}: {metrics['total_trades']} trades, net={metrics['total_net_pnl']}, "
              f"WR={metrics['win_rate']}%, PF={metrics['profit_factor']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "summary.csv", summary_rows)
    if all_trades:
        write_csv(OUT_DIR / "trades.csv", [dataclasses.asdict(t) for t in all_trades])

    diagnostics = {
        "script": "research_asian_session_extreme_reversal.py v2",
        "symbol": SYMBOL, "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "bar_count": len(m1_bars),
        "date_range": {
            "from": m1_bars[0].time.isoformat() if m1_bars else None,
            "to": m1_bars[-1].time.isoformat() if m1_bars else None,
        },
        "variants_tested": len(variants),
    }
    with (OUT_DIR / "diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    print(f"\nResults: {OUT_DIR}")
    print(json.dumps({"summary": summary_rows, "out_dir": str(OUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()
