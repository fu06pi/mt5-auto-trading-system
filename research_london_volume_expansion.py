#!/usr/bin/env python3.14
"""London Open volume-expansion breakout BACKTEST v2 — pullback entry.

v1 showed simple London breakout gets trapped (all variants negative).
v2 tried pullback entry: PF 0.44→0.89 but still negative.
External research confirms: London Breakout on XAUUSD has PF=0.32,
WR=17.5% in 511-trade backtest. Edge has been arbitraged away.

M15 timeframe test found 0 trades in 20000 bars — XAUUSD doesn't
show London open breakouts reliably. Consider H1 EMA Crossover (9/21)
which showed PF=1.18 on XAUUSD per external research.

This script is archived — London Breakout is dead on XAUUSD.
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
OUT_DIR = ROOT / "backtest_reports_london_volume_v2"
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
    lookback_minutes: int = 30
    vol_mult: float = 1.5
    breakout_bars: int = 1
    pullback_max_bars: int = 5
    pullback_level: str = "threshold"             # "threshold" (range h/l) or "midpoint" or "retrace_50pct"
    stop_atr: float = 2.0
    take_profit_atr: float = 4.0
    max_hold_bars: int = 48
    risk_pct: float = 0.0035
    require_close_above_break: bool = True
    london_start_hour: int = 11
    london_start_minute: int = 0
    max_entry_bars: int = 3


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
    pre_vol_avg: float
    breakout_vol: float
    pre_range_points: float
    entry_type: str


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


def backtest(bars: List[Bar], params: Params) -> Tuple[List[Trade], Dict[str, Any]]:
    trades: List[Trade] = []
    last_trade_day: Optional[str] = None

    london_open_mark = params.london_start_hour * 60 + params.london_start_minute
    entry_window_end = london_open_mark + params.max_entry_bars

    for i in range(params.lookback_minutes + 20, len(bars)):
        curr = bars[i]
        bar_min = curr.time.hour * 60 + curr.time.minute
        if not (london_open_mark <= bar_min < entry_window_end):
            continue

        day_key = curr.time.date().isoformat()
        if day_key == last_trade_day:
            continue

        # Pre-London window
        pre_window_start = curr.time - dt.timedelta(minutes=params.lookback_minutes)
        pre_bars = [b for b in bars[:i] if b.time >= pre_window_start]
        if len(pre_bars) < 10:
            continue

        pre_volumes = [b.tick_volume for b in pre_bars]
        vol_avg = mean(pre_volumes)
        if vol_avg <= 0:
            continue

        # Need a spike bar in the London window (which curr is part of)
        if curr.tick_volume < vol_avg * params.vol_mult:
            continue

        pre_high = max(b.high for b in pre_bars)
        pre_low = min(b.low for b in pre_bars)
        pre_range = pre_high - pre_low
        if pre_range <= POINT * 10:
            continue

        if params.require_close_above_break:
            spike_above = curr.close > pre_high
            spike_below = curr.close < pre_low
        else:
            spike_above = curr.high > pre_high
            spike_below = curr.low < pre_low

        if not spike_above and not spike_below:
            continue

        side = "BUY" if spike_above else "SELL"

        # --- ENTRY: pullback instead of initial spike ---
        # Scan forward up to pullback_max_bars from spike bar for pullback
        entry_idx: Optional[int] = None
        entry_price: Optional[float] = None
        entry_type = "spike_close"  # fallback

        for j in range(i, min(i + params.pullback_max_bars + 1, len(bars))):
            pb = bars[j]
            if side == "BUY":
                # Spike above pre_high. Look for pullback to pre_high zone
                threshold = pre_high
                if params.pullback_level == "midpoint":
                    threshold = (pre_high + pre_low) / 2
                elif params.pullback_level == "retrace_50pct":
                    spike_range = pb.high - pre_high
                    threshold = pre_high + spike_range * 0.5

                if pb.close <= threshold and pb.close > pre_low:
                    entry_idx = j
                    entry_price = pb.close
                    entry_type = "pullback"
                    break

                if j == i:
                    continue  # first bar already processed
            else:
                threshold = pre_low
                if params.pullback_level == "midpoint":
                    threshold = (pre_high + pre_low) / 2
                elif params.pullback_level == "retrace_50pct":
                    spike_range = pre_low - pb.low
                    threshold = pre_low - spike_range * 0.5

                if pb.close >= threshold and pb.close < pre_high:
                    entry_idx = j
                    entry_price = pb.close
                    entry_type = "pullback"
                    break

        # If no pullback found, use original spike bar close as entry
        if entry_idx is None:
            entry_idx = i
            entry_price = curr.close
            entry_type = "spike_close"

        atr = atr_from_bars(bars[:entry_idx], 14)
        if atr is None:
            continue

        sl_price = entry_price - params.stop_atr * atr if side == "BUY" else entry_price + params.stop_atr * atr
        tp_price = entry_price + params.take_profit_atr * atr if side == "BUY" else entry_price - params.take_profit_atr * atr

        risk_amount = INITIAL_EQUITY * params.risk_pct
        risk_points = abs(entry_price - sl_price)
        vol = clamp(risk_amount / max(risk_points, POINT) / CONTRACT_SIZE, 0.01, 10.0)

        exit_price = entry_price
        exit_idx = entry_idx
        reason = "max_hold"
        max_exit = min(entry_idx + params.max_hold_bars, len(bars))
        for j in range(entry_idx + 1, max_exit):
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

        bars_held = exit_idx - entry_idx
        if side == "BUY":
            gross_pnl = (exit_price - entry_price) * vol * CONTRACT_SIZE
        else:
            gross_pnl = (entry_price - exit_price) * vol * CONTRACT_SIZE

        spread_cost = SPREAD_POINTS * POINT * vol * CONTRACT_SIZE
        commission = COMMISSION_PER_LOT * vol
        net_pnl = gross_pnl - spread_cost - commission
        r_multiple = gross_pnl / max(risk_amount, 0.01)

        trades.append(Trade(
            params=params.name,
            entry_time=bars[entry_idx].time.isoformat(),
            exit_time=bars[min(exit_idx, len(bars) - 1)].time.isoformat(),
            side=side, entry=entry_price, exit=exit_price,
            sl=sl_price, tp=tp_price, volume=round(vol, 2),
            gross_pnl=round(gross_pnl, 2), net_pnl=round(net_pnl, 2),
            commission=round(commission, 2), spread_cost=round(spread_cost, 2),
            r_multiple=round(r_multiple, 2), reason=reason, bars_held=bars_held,
            atr=round(atr, 2), pre_vol_avg=round(vol_avg, 1),
            breakout_vol=curr.tick_volume, pre_range_points=round(pre_range / POINT, 1),
            entry_type=entry_type,
        ))
        last_trade_day = day_key

    if not trades:
        return [], {"total_trades": 0, "total_net_pnl": 0.0}

    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]
    total_gross = sum(t.gross_pnl for t in trades)
    total_net = sum(t.net_pnl for t in trades)
    gross_wins = sum(t.gross_pnl for t in wins)
    gross_losses = abs(sum(t.gross_pnl for t in losses))
    pf = gross_wins / max(gross_losses, 0.01)

    pullbacks = [t for t in trades if t.entry_type == "pullback"]
    spikes = [t for t in trades if t.entry_type == "spike_close"]

    metrics = {
        "total_trades": len(trades),
        "win_count": len(wins), "loss_count": len(losses),
        "win_rate": round(len(wins) / max(len(trades), 1) * 100, 1),
        "total_gross_pnl": round(total_gross, 2),
        "total_net_pnl": round(total_net, 2),
        "profit_factor": round(pf, 2),
        "avg_r_multiple": round(mean([t.r_multiple for t in trades]), 2),
        "avg_bars_held": round(mean([t.bars_held for t in trades]), 1),
        "pullback_trades": len(pullbacks),
        "pullback_net": round(sum(t.net_pnl for t in pullbacks), 2),
        "spike_trades": len(spikes),
        "spike_net": round(sum(t.net_pnl for t in spikes), 2),
    }
    return trades, metrics


def main() -> None:
    print(f"Fetching {MAX_M1_BARS} M1 bars...")
    m1_bars = fetch_bars("M1", MAX_M1_BARS)
    print(f"Got {len(m1_bars)} bars ({m1_bars[0].time.date()} to {m1_bars[-1].time.date()}).")

    variants = [
        Params(name="base_pull_threshold", vol_mult=1.5, pullback_level="threshold"),
        Params(name="pull_midpoint", vol_mult=1.5, pullback_level="midpoint"),
        Params(name="pull_retrace50", vol_mult=1.5, pullback_level="retrace_50pct"),
        Params(name="tight_2v_pull_threshold", vol_mult=2.0, pullback_level="threshold"),
        Params(name="long_pb5_pull_threshold", vol_mult=1.5, pullback_level="threshold", pullback_max_bars=5),
        Params(name="short_pb2_pull_threshold", vol_mult=1.5, pullback_level="threshold", pullback_max_bars=2),
        Params(name="base_close_free", vol_mult=1.5, pullback_level="threshold", require_close_above_break=False),
        Params(name="aggro_1.3v_pull_threshold_1.8s_3.5tp", vol_mult=1.3,
               pullback_level="threshold", stop_atr=1.8, take_profit_atr=3.5),
        Params(name="spike_only_v1_compat", vol_mult=1.5, pullback_max_bars=0,
               pullback_level="threshold"),
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
        "script": "research_london_volume_expansion.py v2",
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
