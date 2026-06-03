#!/usr/bin/env python3
"""Spike: XAUUSD M5 false-breakout sharp reversal signal.

Read-only research script. It tests wick/reclaim false breakouts against recent range extremes
and reports R-multiple outcomes. No MT5 orders, no live state changes.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class Variant:
    name: str
    lookback: int
    break_atr: float
    close_back_atr: float
    wick_ratio: float
    max_hold: int
    rr: float
    trend_mode: str  # any, counter_htf, with_htf


def load_bars(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    time_col = cols.get("time")
    if time_col is None:
        raise ValueError("missing time column")
    df = df.rename(columns={cols.get("open", "Open"): "open", cols.get("high", "High"): "high", cols.get("low", "Low"): "low", cols.get("close", "Close"): "close"})
    df["time"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df[["time", "open", "high", "low", "close"]].dropna().sort_values("time").reset_index(drop=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    # M5 equivalent of H1 SMA50/200: 600 / 2400 M5 bars.
    df["h1_fast_proxy"] = df["close"].rolling(600).mean()
    df["h1_slow_proxy"] = df["close"].rolling(2400).mean()
    df["htf"] = "NEUTRAL"
    df.loc[df["h1_fast_proxy"] > df["h1_slow_proxy"], "htf"] = "BULL"
    df.loc[df["h1_fast_proxy"] < df["h1_slow_proxy"], "htf"] = "BEAR"
    return df


def iter_signals(df: pd.DataFrame, v: Variant) -> List[Dict[str, object]]:
    signals: List[Dict[str, object]] = []
    for i in range(max(v.lookback + 1, 2500), len(df) - v.max_hold - 2):
        row = df.iloc[i]
        atr = float(row.atr14)
        if not atr or pd.isna(atr):
            continue
        prev = df.iloc[i - v.lookback:i]
        prior_high = float(prev.high.max())
        prior_low = float(prev.low.min())
        body_high = max(float(row.open), float(row.close))
        body_low = min(float(row.open), float(row.close))
        rng = max(float(row.high - row.low), 1e-9)
        upper_wick = float(row.high - body_high) / rng
        lower_wick = float(body_low - row.low) / rng
        htf = str(row.htf)

        # Upthrust: price pierces range high, closes back inside -> SELL reversal.
        up_break = float(row.high) >= prior_high + v.break_atr * atr
        up_reclaim_fail = float(row.close) <= prior_high - v.close_back_atr * atr
        if up_break and up_reclaim_fail and upper_wick >= v.wick_ratio:
            if v.trend_mode == "with_htf" and htf != "BEAR":
                pass
            elif v.trend_mode == "counter_htf" and htf != "BULL":
                pass
            else:
                signals.append({"i": i, "time": row.time, "dir": "SELL", "extreme": float(row.high), "atr": atr, "htf": htf})

        # Spring: price pierces range low, closes back inside -> BUY reversal.
        dn_break = float(row.low) <= prior_low - v.break_atr * atr
        dn_reclaim = float(row.close) >= prior_low + v.close_back_atr * atr
        if dn_break and dn_reclaim and lower_wick >= v.wick_ratio:
            if v.trend_mode == "with_htf" and htf != "BULL":
                pass
            elif v.trend_mode == "counter_htf" and htf != "BEAR":
                pass
            else:
                signals.append({"i": i, "time": row.time, "dir": "BUY", "extreme": float(row.low), "atr": atr, "htf": htf})
    return signals


def simulate(df: pd.DataFrame, signals: List[Dict[str, object]], v: Variant) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    last_exit_i = -1
    for sig in signals:
        i = int(sig["i"])
        if i <= last_exit_i:
            continue
        entry_i = i + 1
        entry = float(df.iloc[entry_i].open)
        atr = float(sig["atr"])
        if sig["dir"] == "SELL":
            sl = max(float(sig["extreme"]), entry) + 0.10 * atr
            risk = sl - entry
            if risk <= 0:
                continue
            tp = entry - v.rr * risk
        else:
            sl = min(float(sig["extreme"]), entry) - 0.10 * atr
            risk = entry - sl
            if risk <= 0:
                continue
            tp = entry + v.rr * risk
        outcome = 0.0
        exit_i = entry_i + v.max_hold
        exit_price = float(df.iloc[exit_i].close)
        reason = "TIME"
        for j in range(entry_i, min(len(df), entry_i + v.max_hold + 1)):
            hi = float(df.iloc[j].high)
            lo = float(df.iloc[j].low)
            if sig["dir"] == "SELL":
                # Conservative: SL first if both touched in same bar.
                if hi >= sl:
                    outcome, exit_i, exit_price, reason = -1.0, j, sl, "SL"
                    break
                if lo <= tp:
                    outcome, exit_i, exit_price, reason = v.rr, j, tp, "TP"
                    break
            else:
                if lo <= sl:
                    outcome, exit_i, exit_price, reason = -1.0, j, sl, "SL"
                    break
                if hi >= tp:
                    outcome, exit_i, exit_price, reason = v.rr, j, tp, "TP"
                    break
        else:
            if sig["dir"] == "SELL":
                outcome = (entry - exit_price) / risk
            else:
                outcome = (exit_price - entry) / risk
        last_exit_i = exit_i
        rows.append({
            "time": sig["time"], "dir": sig["dir"], "htf": sig["htf"], "entry": entry,
            "sl": sl, "tp": tp, "exit_price": exit_price, "exit_reason": reason,
            "bars_held": exit_i - entry_i, "r": outcome,
        })
    return pd.DataFrame(rows)


def max_drawdown_r(rs: pd.Series) -> float:
    if rs.empty:
        return 0.0
    eq = rs.cumsum()
    peak = eq.cummax()
    return float((eq - peak).min())


def summarize(trades: pd.DataFrame) -> Dict[str, object]:
    if trades.empty:
        return {"trades": 0}
    wins = trades[trades.r > 0]
    losses = trades[trades.r < 0]
    return {
        "trades": int(len(trades)),
        "win_rate": round(float(len(wins) / len(trades) * 100), 2),
        "total_r": round(float(trades.r.sum()), 2),
        "avg_r": round(float(trades.r.mean()), 3),
        "median_r": round(float(trades.r.median()), 3),
        "max_dd_r": round(max_drawdown_r(trades.r), 2),
        "tp": int((trades.exit_reason == "TP").sum()),
        "sl": int((trades.exit_reason == "SL").sum()),
        "time_exit": int((trades.exit_reason == "TIME").sum()),
        "buy_trades": int((trades.dir == "BUY").sum()),
        "sell_trades": int((trades.dir == "SELL").sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="/home/chain4655/Documents/backtest_reports/xauusd_5y_weekly_reset/data/XAUUSD_M5_5y_mt5.csv")
    parser.add_argument("--out", default="/home/chain4655/Documents/Projects/MT5/spikes/001-false-breakout-reversal/results")
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = add_indicators(load_bars(Path(args.csv)))
    variants = [
        Variant("strict_any_2r", 20, 0.15, 0.05, 0.45, 18, 2.0, "any"),
        Variant("strict_counter_htf_2r", 20, 0.15, 0.05, 0.45, 18, 2.0, "counter_htf"),
        Variant("strict_with_htf_2r", 20, 0.15, 0.05, 0.45, 18, 2.0, "with_htf"),
        Variant("sharp_with_htf_2r", 30, 0.20, 0.10, 0.55, 18, 2.0, "with_htf"),
        Variant("sharp_with_htf_1p5r", 30, 0.20, 0.10, 0.55, 12, 1.5, "with_htf"),
        Variant("very_sharp_with_htf_2r", 40, 0.25, 0.12, 0.65, 18, 2.0, "with_htf"),
        Variant("strict_with_htf_3r", 20, 0.15, 0.05, 0.45, 30, 3.0, "with_htf"),
        Variant("loose_any_1p5r", 12, 0.05, 0.00, 0.35, 12, 1.5, "any"),
        Variant("loose_counter_htf_1p5r", 12, 0.05, 0.00, 0.35, 12, 1.5, "counter_htf"),
        Variant("tight_fast_any_1r", 10, 0.05, 0.00, 0.40, 6, 1.0, "any"),
    ]
    summary_rows = []
    for v in variants:
        sigs = iter_signals(df, v)
        trades = simulate(df, sigs, v)
        trades.to_csv(out / f"{v.name}_trades.csv", index=False)
        row = {"variant": v.name, **summarize(trades)}
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["total_r", "avg_r"], ascending=False)
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
