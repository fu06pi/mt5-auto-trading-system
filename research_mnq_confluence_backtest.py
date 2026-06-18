#!/usr/bin/env python3
"""Research-only MNQ/NQ 5m confluence pullback strategy prototype.

Signal inspired by: retracement 0.5-0.618 + VWAP + line-chart trendline break/retest
+ support/resistance flip. This script does NOT touch MT5/live state.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yfinance as yf


OUT_ROOT = Path("/home/chain4655/Documents/backtest_reports/mnq_confluence_pullback")


@dataclasses.dataclass(frozen=True)
class Params:
    name: str
    rr: float = 2.0
    max_hold_bars: int = 72
    setup_lookback: int = 96
    trend_lookback: int = 36
    min_impulse_atr: float = 2.0
    atr_stop_mult: float = 0.35
    vwap_tolerance_atr: float = 0.45
    trendline_tolerance_atr: float = 0.55
    sr_tolerance_atr: float = 0.60
    require_vwap: bool = True
    require_sr_flip: bool = True
    require_trend_retest: bool = True
    session_start_hour: int = 14  # UTC, roughly US cash open window start
    session_end_hour: int = 21
    fee_points_roundtrip: float = 1.2  # conservative MNQ all-in friction in index points
    initial_equity: float = 100_000.0
    risk_pct: float = 0.0025
    daily_dd_stop: float = 0.03
    total_dd_stop: float = 0.08


@dataclasses.dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    entry: float
    stop: float
    target: float
    exit: float
    pnl_points: float
    r_multiple: float
    reason: str
    setup_score: int


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]).lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]
    return df


def load_data(symbol: str, period: str, interval: str, cache: Path) -> pd.DataFrame:
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["datetime"])
        df = df.set_index("datetime")
    else:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
        if df.empty:
            raise RuntimeError(f"No data downloaded for {symbol} {period} {interval}")
        df = _flatten_columns(df)
        df.index.name = "datetime"
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.reset_index().to_csv(cache, index=False)
    df = _flatten_columns(df.copy())
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")
    df = df.dropna(subset=["open", "high", "low", "close"])
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr14"] = tr.rolling(14).mean()

    session = out.index.tz_convert("America/New_York").date
    typical = (out["high"] + out["low"] + out["close"]) / 3.0
    vol = out["volume"].replace(0, 1)
    out["vwap"] = (typical * vol).groupby(session).cumsum() / vol.groupby(session).cumsum()
    out["ny_hour"] = out.index.hour
    return out


def regression_line(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    if n < 2:
        return 0.0, values[-1] if values else 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    denom = sum((i - x_mean) ** 2 for i in range(n))
    if denom == 0:
        return 0.0, values[-1]
    slope = sum((i - x_mean) * (values[i] - y_mean) for i in range(n)) / denom
    intercept = y_mean - slope * x_mean
    return slope, intercept


def in_session(ts: pd.Timestamp, p: Params) -> bool:
    return p.session_start_hour <= ts.hour <= p.session_end_hour


def find_signal(df: pd.DataFrame, i: int, p: Params) -> Optional[Dict[str, float]]:
    if i < max(p.setup_lookback, p.trend_lookback) + 3:
        return None
    row = df.iloc[i]
    if not in_session(df.index[i], p) or not math.isfinite(row.atr14) or row.atr14 <= 0:
        return None

    look = df.iloc[i - p.setup_lookback : i]
    swing_low = float(look["low"].min())
    swing_high = float(look["high"].max())
    low_pos = int(look["low"].values.argmin())
    high_pos = int(look["high"].values.argmax())
    if low_pos >= high_pos:
        return None
    impulse = swing_high - swing_low
    if impulse < p.min_impulse_atr * row.atr14:
        return None

    fib_50 = swing_high - 0.500 * impulse
    fib_618 = swing_high - 0.618 * impulse
    zone_low = min(fib_50, fib_618)
    zone_high = max(fib_50, fib_618)
    in_fib_zone = row.low <= zone_high and row.high >= zone_low
    if not in_fib_zone:
        return None

    pre_breakout = df.iloc[i - p.trend_lookback : i]
    slope, intercept = regression_line(pre_breakout["close"].tolist())
    line_now = intercept + slope * (p.trend_lookback - 1)
    prior_line = intercept + slope * (p.trend_lookback - 3)
    broke_downtrend = slope < 0 and df.iloc[i - 2].close > prior_line
    trend_retest = abs(float(row.low) - line_now) <= p.trendline_tolerance_atr * row.atr14
    if p.require_trend_retest and not (broke_downtrend and trend_retest):
        return None

    vwap_ok = abs(float(row.low) - float(row.vwap)) <= p.vwap_tolerance_atr * row.atr14
    if p.require_vwap and not vwap_ok:
        return None

    breakout_level = float(df.iloc[i - p.trend_lookback : i - 2]["high"].max())
    sr_flip = abs(float(row.low) - breakout_level) <= p.sr_tolerance_atr * row.atr14
    if p.require_sr_flip and not sr_flip:
        return None

    bullish_reject = row.close > row.open and row.close > (row.low + 0.60 * (row.high - row.low))
    if not bullish_reject:
        return None

    score = 1 + int(vwap_ok) + int(broke_downtrend and trend_retest) + int(sr_flip)
    # Day-trade prototype: invalidate below the actual confluence/rejection bar,
    # not below the whole impulse swing low, otherwise RR targets become unrealistic.
    stop = float(row.low) - p.atr_stop_mult * row.atr14
    entry = float(df.iloc[i + 1].open) if i + 1 < len(df) else float(row.close)
    if entry <= stop:
        return None
    risk = entry - stop
    target = entry + p.rr * risk
    return {"entry": entry, "stop": stop, "target": target, "score": float(score)}


def backtest(df: pd.DataFrame, p: Params) -> Tuple[List[Trade], Dict[str, float]]:
    trades: List[Trade] = []
    equity = p.initial_equity
    peak_equity = equity
    day_start = equity
    current_day = None
    stopped = False
    i = 0
    while i < len(df) - 2 and not stopped:
        ts = df.index[i]
        day = ts.date()
        if day != current_day:
            current_day = day
            day_start = equity
        if (day_start - equity) / day_start >= p.daily_dd_stop:
            # skip until next session/day
            i += 1
            continue
        sig = find_signal(df, i, p)
        if sig is None:
            i += 1
            continue
        entry_i = i + 1
        entry = sig["entry"]
        stop = sig["stop"]
        target = sig["target"]
        risk = entry - stop
        exit_price = float(df.iloc[min(entry_i + p.max_hold_bars, len(df) - 1)].close)
        exit_i = min(entry_i + p.max_hold_bars, len(df) - 1)
        reason = "max_hold"
        for j in range(entry_i, min(entry_i + p.max_hold_bars + 1, len(df))):
            bar = df.iloc[j]
            # Conservative: same-bar stop before target for long.
            if float(bar.low) <= stop:
                exit_price = stop
                exit_i = j
                reason = "stop"
                break
            if float(bar.high) >= target:
                exit_price = target
                exit_i = j
                reason = "target"
                break
        gross_points = exit_price - entry
        pnl_points = gross_points - p.fee_points_roundtrip
        r_multiple = pnl_points / risk if risk > 0 else 0.0
        risk_dollars = equity * p.risk_pct
        pnl_dollars = risk_dollars * r_multiple
        equity += pnl_dollars
        peak_equity = max(peak_equity, equity)
        trades.append(
            Trade(
                entry_time=df.index[entry_i],
                exit_time=df.index[exit_i],
                side="LONG",
                entry=entry,
                stop=stop,
                target=target,
                exit=exit_price,
                pnl_points=pnl_points,
                r_multiple=r_multiple,
                reason=reason,
                setup_score=int(sig["score"]),
            )
        )
        if (p.initial_equity - equity) / p.initial_equity >= p.total_dd_stop:
            stopped = True
        i = max(exit_i + 1, i + 1)
    metrics = summarize(trades, p, equity)
    return trades, metrics


def summarize(trades: List[Trade], p: Params, final_equity: float) -> Dict[str, Any]:
    if not trades:
        return {
            "name": p.name,
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_r": 0.0,
            "avg_r": 0.0,
            "final_equity": final_equity,
            "return_pct": (final_equity / p.initial_equity - 1.0) * 100.0,
            "max_dd_pct": 0.0,
        }
    rs = [t.r_multiple for t in trades]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    eq = p.initial_equity
    peak = eq
    max_dd = 0.0
    for r in rs:
        eq += eq * p.risk_pct * r
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak if peak else 0.0)
    return {
        "name": p.name,
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) * 100.0,
        "profit_factor": gross_win / gross_loss if gross_loss else 999.0,
        "net_r": sum(rs),
        "avg_r": sum(rs) / len(rs),
        "final_equity": eq,
        "return_pct": (eq / p.initial_equity - 1.0) * 100.0,
        "max_dd_pct": max_dd * 100.0,
        "targets": sum(1 for t in trades if t.reason == "target"),
        "stops": sum(1 for t in trades if t.reason == "stop"),
        "max_holds": sum(1 for t in trades if t.reason == "max_hold"),
    }


def variants() -> Iterable[Params]:
    base = Params(name="strict_rr2")
    yield base
    yield dataclasses.replace(base, name="strict_rr3", rr=3.0)
    yield dataclasses.replace(base, name="no_sr_rr2", require_sr_flip=False)
    yield dataclasses.replace(base, name="no_vwap_rr2", require_vwap=False)
    yield dataclasses.replace(base, name="loose_rr2", require_sr_flip=False, vwap_tolerance_atr=0.75, trendline_tolerance_atr=0.80)
    yield dataclasses.replace(base, name="loose_rr3", rr=3.0, require_sr_flip=False, vwap_tolerance_atr=0.75, trendline_tolerance_atr=0.80)


def save_outputs(out_dir: Path, df: pd.DataFrame, all_metrics: List[Dict[str, Any]], trade_map: Dict[str, List[Trade]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_metrics).sort_values(["profit_factor", "net_r"], ascending=False).to_csv(
        out_dir / "summary.csv", index=False
    )
    for name, trades in trade_map.items():
        pd.DataFrame([dataclasses.asdict(t) for t in trades]).to_csv(out_dir / f"{name}_trades.csv", index=False)
    meta = {
        "bars": len(df),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "note": "Research-only yfinance MNQ=F 5m prototype. No live/active_plan changes.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="MNQ=F")
    parser.add_argument("--period", default="60d")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else OUT_ROOT / pd.Timestamp.now('UTC').strftime("%Y%m%d_%H%M%S")
    cache = OUT_ROOT / "cache" / f"{args.symbol.replace('=', '')}_{args.period}_{args.interval}.csv"
    df = add_indicators(load_data(args.symbol, args.period, args.interval, cache))
    all_metrics: List[Dict[str, Any]] = []
    trade_map: Dict[str, List[Trade]] = {}
    for p in variants():
        trades, metrics = backtest(df, p)
        all_metrics.append(metrics)
        trade_map[p.name] = trades
    save_outputs(out_dir, df, all_metrics, trade_map)
    print(f"DATA {args.symbol} {args.period} {args.interval}: {df.index.min()} -> {df.index.max()} bars={len(df)}")
    print(f"OUT {out_dir}")
    print(pd.DataFrame(all_metrics).sort_values(["profit_factor", "net_r"], ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
