#!/usr/bin/env python3
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    from pymt5linux import MetaTrader5
except ImportError:
    sys.path.insert(0, "/home/chain4655/.openharness-venv/lib64/python3.14/site-packages")
    from pymt5linux import MetaTrader5

ROOT = Path("/home/chain4655/Documents/Projects/MT5")
OUT_ROOT = Path("/home/chain4655/Documents/backtest_reports/twitter_xauusd_candidates")
SYMBOL = "XAUUSD"
INITIAL_EQUITY = 10000.0
CONTRACT_SIZE = 100.0
MIN_LOT = 0.01
MAX_LOT = 1.2
RISK_PCT = 0.005
COMMISSION_PER_LOT_ROUND = 6.0
SPREAD_PRICE_COST = 0.40  # current live spread often ~40 points = 0.40 XAUUSD


@dataclasses.dataclass(frozen=True)
class Trade:
    strategy: str
    period: str
    entry_time: dt.datetime
    exit_time: dt.datetime
    side: str
    entry: float
    exit: float
    sl: float
    tp: float
    lots: float
    gross_pnl: float
    commission: float
    spread_cost: float
    net_pnl: float
    r_multiple: float
    reason: str
    bars_held: int
    notes: str


def timeframe_minutes(tf: int) -> object:
    return tf


def fetch_bars(days: int = 190, timeframe: str = "M5") -> pd.DataFrame:
    mt5 = MetaTrader5(host="127.0.0.1", port=18812)
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        tf = mt5.TIMEFRAME_M5 if timeframe == "M5" else mt5.TIMEFRAME_M15
        end = dt.datetime.now()
        start = end - dt.timedelta(days=days)
        rates = mt5.copy_rates_range(SYMBOL, tf, start, end)
        if rates is None or len(rates) < 1000:
            count = int(days * (288 if timeframe == "M5" else 96) * 1.2)
            rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, count)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No rates returned: {mt5.last_error()}")
        df = pd.DataFrame(rates)
        # pymt5linux returns a numpy structured array; keep field names intact.
        if "time" not in df.columns:
            df = pd.DataFrame.from_records(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={"tick_volume": "volume"})
        df = df[["time", "open", "high", "low", "close", "volume"]].copy()
        df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
        return df
    finally:
        mt5.shutdown()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - prev_close).abs(),
        (out["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["atr14"] = tr.rolling(14).mean()
    out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["ema50"] = out["close"].ewm(span=50, adjust=False).mean()
    out["date"] = out["time"].dt.date
    out["hour"] = out["time"].dt.hour
    out["minute"] = out["time"].dt.minute
    out["dow"] = out["time"].dt.dayofweek
    return out


def position_size(equity: float, entry: float, sl: float) -> float:
    risk_amount = equity * RISK_PCT
    risk_per_lot = abs(entry - sl) * CONTRACT_SIZE
    if risk_per_lot <= 0:
        return 0.0
    lots = math.floor(min(MAX_LOT, risk_amount / risk_per_lot) / MIN_LOT) * MIN_LOT
    return max(MIN_LOT, round(lots, 2)) if lots >= MIN_LOT else 0.0


def make_trade(strategy: str, period: str, side: str, entry_time: dt.datetime, exit_time: dt.datetime,
               entry: float, exit_price: float, sl: float, tp: float, lots: float,
               reason: str, bars_held: int, notes: str) -> Trade:
    mult = 1.0 if side == "BUY" else -1.0
    gross = (exit_price - entry) * mult * lots * CONTRACT_SIZE
    commission = -COMMISSION_PER_LOT_ROUND * lots
    spread_cost = -SPREAD_PRICE_COST * lots * CONTRACT_SIZE
    net = gross + commission + spread_cost
    risk_amount = abs(entry - sl) * lots * CONTRACT_SIZE
    return Trade(
        strategy=strategy,
        period=period,
        entry_time=entry_time,
        exit_time=exit_time,
        side=side,
        entry=round(entry, 2),
        exit=round(exit_price, 2),
        sl=round(sl, 2),
        tp=round(tp, 2),
        lots=lots,
        gross_pnl=round(gross, 2),
        commission=round(commission, 2),
        spread_cost=round(spread_cost, 2),
        net_pnl=round(net, 2),
        r_multiple=round(net / max(risk_amount, 1e-9), 3),
        reason=reason,
        bars_held=bars_held,
        notes=notes,
    )


def simulate_entries(df: pd.DataFrame, period: str, strategy: str,
                     entries: List[Dict[str, object]], max_hold_bars: int = 36) -> List[Trade]:
    trades: List[Trade] = []
    equity = INITIAL_EQUITY
    last_exit_idx = -1
    for e in sorted(entries, key=lambda x: int(x["idx"])):
        idx = int(e["idx"])
        if idx <= last_exit_idx or idx >= len(df) - 2:
            continue
        side = str(e["side"])
        entry_idx = idx + 1
        entry_bar = df.iloc[entry_idx]
        entry = float(entry_bar.open)
        sl = float(e["sl"])
        tp = float(e["tp"])
        if side == "BUY" and not (sl < entry < tp):
            continue
        if side == "SELL" and not (tp < entry < sl):
            continue
        lots = position_size(equity, entry, sl)
        if lots <= 0:
            continue
        exit_price = float(df.iloc[min(entry_idx + max_hold_bars, len(df) - 1)].close)
        exit_idx = min(entry_idx + max_hold_bars, len(df) - 1)
        reason = "TIME"
        for j in range(entry_idx, min(entry_idx + max_hold_bars + 1, len(df))):
            bar = df.iloc[j]
            if side == "BUY":
                hit_sl = float(bar.low) <= sl
                hit_tp = float(bar.high) >= tp
            else:
                hit_sl = float(bar.high) >= sl
                hit_tp = float(bar.low) <= tp
            if hit_sl and hit_tp:
                exit_price, exit_idx, reason = sl, j, "SL_FIRST"
                break
            if hit_sl:
                exit_price, exit_idx, reason = sl, j, "SL"
                break
            if hit_tp:
                exit_price, exit_idx, reason = tp, j, "TP"
                break
        t = make_trade(
            strategy=strategy,
            period=period,
            side=side,
            entry_time=entry_bar.time.to_pydatetime(),
            exit_time=df.iloc[exit_idx].time.to_pydatetime(),
            entry=entry,
            exit_price=exit_price,
            sl=sl,
            tp=tp,
            lots=lots,
            reason=reason,
            bars_held=exit_idx - entry_idx,
            notes=str(e.get("notes", "")),
        )
        trades.append(t)
        equity += t.net_pnl
        last_exit_idx = exit_idx + 3
    return trades


def asian_london_breakout(df: pd.DataFrame) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    for day, g in df.groupby("date", sort=True):
        asia = g[(g.hour >= 0) & (g.hour < 7)]
        london = g[(g.hour >= 7) & (g.hour < 11)]
        if len(asia) < 24 or london.empty:
            continue
        hi, lo = float(asia.high.max()), float(asia.low.min())
        rng = hi - lo
        if rng < 2.0 or rng > 35.0:
            continue
        for idx, row in london.iterrows():
            atr = float(row.atr14) if not math.isnan(float(row.atr14)) else rng / 2
            prev = df.iloc[idx - 1] if idx > 0 else row
            if row.close > hi and prev.close <= hi:
                sl = min(lo, float(row.close) - max(atr * 1.2, rng * 0.55))
                tp = float(row.close) + (float(row.close) - sl) * 2.0
                entries.append({"idx": idx, "side": "BUY", "sl": sl, "tp": tp, "notes": f"asia_hi={hi:.2f};asia_lo={lo:.2f}"})
                break
            if row.close < lo and prev.close >= lo:
                sl = max(hi, float(row.close) + max(atr * 1.2, rng * 0.55))
                tp = float(row.close) - (sl - float(row.close)) * 2.0
                entries.append({"idx": idx, "side": "SELL", "sl": sl, "tp": tp, "notes": f"asia_hi={hi:.2f};asia_lo={lo:.2f}"})
                break
    return entries


def pdh_pdl_sweep_reversal(df: pd.DataFrame) -> List[Dict[str, object]]:
    daily = df.groupby("date").agg(high=("high", "max"), low=("low", "min")).shift(1)
    entries: List[Dict[str, object]] = []
    for idx, row in df.iterrows():
        if row.hour < 7 or row.hour > 18 or row.date not in daily.index:
            continue
        prev = daily.loc[row.date]
        if pd.isna(prev.high) or pd.isna(prev.low):
            continue
        atr = float(row.atr14)
        if math.isnan(atr) or atr <= 0:
            continue
        # Sweep previous high and close back below -> short.
        if row.high > float(prev.high) + 0.05 and row.close < float(prev.high):
            sl = float(row.high) + atr * 0.35
            tp = float(row.close) - (sl - float(row.close)) * 2.0
            entries.append({"idx": idx, "side": "SELL", "sl": sl, "tp": tp, "notes": f"sweep_PDH={float(prev.high):.2f}"})
        # Sweep previous low and close back above -> long.
        elif row.low < float(prev.low) - 0.05 and row.close > float(prev.low):
            sl = float(row.low) - atr * 0.35
            tp = float(row.close) + (float(row.close) - sl) * 2.0
            entries.append({"idx": idx, "side": "BUY", "sl": sl, "tp": tp, "notes": f"sweep_PDL={float(prev.low):.2f}"})
    return entries


def us_orb(df: pd.DataFrame) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    for day, g in df.groupby("date", sort=True):
        # Server-time proxy used in prior reports: 13:30-16:00 / 14:00-17:00 style US window.
        orb = g[((g.hour == 13) & (g.minute >= 30)) | ((g.hour == 14) & (g.minute < 30))]
        trade_window = g[(g.hour >= 14) & (g.hour < 17)]
        if len(orb) < 8 or trade_window.empty:
            continue
        hi, lo = float(orb.high.max()), float(orb.low.min())
        rng = hi - lo
        if rng < 1.5 or rng > 25.0:
            continue
        for idx, row in trade_window.iterrows():
            prev = df.iloc[idx - 1] if idx > 0 else row
            if row.close > hi and prev.close <= hi:
                sl = lo
                tp = float(row.close) + (float(row.close) - sl) * 1.8
                entries.append({"idx": idx, "side": "BUY", "sl": sl, "tp": tp, "notes": f"us_orb_hi={hi:.2f};lo={lo:.2f}"})
                break
            if row.close < lo and prev.close >= lo:
                sl = hi
                tp = float(row.close) - (sl - float(row.close)) * 1.8
                entries.append({"idx": idx, "side": "SELL", "sl": sl, "tp": tp, "notes": f"us_orb_hi={hi:.2f};lo={lo:.2f}"})
                break
    return entries


def false_breakout_reversal(df: pd.DataFrame) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    rolling_hi = df.high.shift(1).rolling(36).max()
    rolling_lo = df.low.shift(1).rolling(36).min()
    for idx, row in df.iterrows():
        if idx < 60 or row.hour < 6 or row.hour > 18:
            continue
        hi = float(rolling_hi.iloc[idx])
        lo = float(rolling_lo.iloc[idx])
        atr = float(row.atr14)
        if math.isnan(hi) or math.isnan(lo) or math.isnan(atr) or atr <= 0:
            continue
        rng = hi - lo
        if rng > atr * 5.0 or rng < atr * 1.2:
            continue
        if row.high > hi and row.close < hi:
            sl = float(row.high) + atr * 0.25
            tp = float(row.close) - (sl - float(row.close)) * 1.8
            entries.append({"idx": idx, "side": "SELL", "sl": sl, "tp": tp, "notes": f"false_break_hi={hi:.2f}"})
        elif row.low < lo and row.close > lo:
            sl = float(row.low) - atr * 0.25
            tp = float(row.close) + (float(row.close) - sl) * 1.8
            entries.append({"idx": idx, "side": "BUY", "sl": sl, "tp": tp, "notes": f"false_break_lo={lo:.2f}"})
    return entries


def ma_pullback(df: pd.DataFrame) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    for idx, row in df.iterrows():
        if idx < 80 or row.hour < 7 or row.hour > 19:
            continue
        atr = float(row.atr14)
        if math.isnan(atr) or atr <= 0:
            continue
        prev = df.iloc[idx - 1]
        uptrend = float(row.ema20) > float(row.ema50) and row.close > row.ema50
        downtrend = float(row.ema20) < float(row.ema50) and row.close < row.ema50
        # Touch EMA20 then close back with trend.
        if uptrend and row.low <= row.ema20 <= row.high and row.close > row.open and prev.close < prev.ema20:
            sl = float(row.low) - atr * 0.5
            tp = float(row.close) + (float(row.close) - sl) * 1.8
            entries.append({"idx": idx, "side": "BUY", "sl": sl, "tp": tp, "notes": "ema20_pullback_uptrend"})
        elif downtrend and row.low <= row.ema20 <= row.high and row.close < row.open and prev.close > prev.ema20:
            sl = float(row.high) + atr * 0.5
            tp = float(row.close) - (sl - float(row.close)) * 1.8
            entries.append({"idx": idx, "side": "SELL", "sl": sl, "tp": tp, "notes": "ema20_pullback_downtrend"})
    return entries


def metrics(trades: List[Trade]) -> Dict[str, object]:
    equity = INITIAL_EQUITY
    peak = equity
    dd = 0.0
    eq_curve = []
    for t in sorted(trades, key=lambda x: x.exit_time):
        equity += t.net_pnl
        peak = max(peak, equity)
        dd = min(dd, equity / peak - 1.0)
        eq_curve.append({"time": t.exit_time.isoformat(), "equity": round(equity, 2), "dd_pct": round((equity / peak - 1.0) * 100, 3)})
    wins = [t.net_pnl for t in trades if t.net_pnl > 0]
    losses = [t.net_pnl for t in trades if t.net_pnl <= 0]
    gp = sum(wins)
    gl = -sum(losses)
    return {
        "trades": len(trades),
        "net_pnl": round(equity - INITIAL_EQUITY, 2),
        "return_pct": round((equity / INITIAL_EQUITY - 1.0) * 100, 3),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 2) if trades else 0.0,
        "profit_factor": round(gp / gl, 3) if gl > 0 else (999.0 if gp > 0 else 0.0),
        "max_dd_pct": round(dd * 100, 3),
        "expectancy_usd": round(statistics.fmean([t.net_pnl for t in trades]), 2) if trades else 0.0,
        "avg_r": round(statistics.fmean([t.r_multiple for t in trades]), 3) if trades else 0.0,
        "gross_profit": round(gp, 2),
        "gross_loss": round(-gl, 2),
        "equity_curve": eq_curve,
    }


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fields: Optional[List[str]] = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = add_indicators(fetch_bars(days=190, timeframe="M5"))
    raw.to_csv(out_dir / "input_xauusd_m5.csv", index=False)

    periods = {"3m": 92, "6m": 184}
    strategies: Dict[str, Tuple[Callable[[pd.DataFrame], List[Dict[str, object]]], int]] = {
        "london_asian_range_breakout": (asian_london_breakout, 48),
        "pdh_pdl_sweep_reversal": (pdh_pdl_sweep_reversal, 48),
        "us_session_orb": (us_orb, 36),
        "range_false_breakout_reversal": (false_breakout_reversal, 36),
        "ema20_pullback_trend": (ma_pullback, 36),
    }
    all_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    summary_json: Dict[str, object] = {
        "generated_at": dt.datetime.now().isoformat(sep=" "),
        "symbol": SYMBOL,
        "timeframe": "M5",
        "source": "MT5 copy_rates via pymt5linux",
        "input_start": raw.time.iloc[0].isoformat(),
        "input_end": raw.time.iloc[-1].isoformat(),
        "assumptions": {
            "initial_equity": INITIAL_EQUITY,
            "risk_pct": RISK_PCT,
            "max_lots": MAX_LOT,
            "commission_per_lot_round_usd": COMMISSION_PER_LOT_ROUND,
            "spread_price_cost": SPREAD_PRICE_COST,
            "fill_model": "next-bar open entry; SL priority if SL and TP hit in same bar",
        },
        "strategies": {},
    }
    curve_rows: List[Dict[str, object]] = []
    for period, days in periods.items():
        cutoff = raw.time.max() - pd.Timedelta(days=days)
        df = raw[raw.time >= cutoff].reset_index(drop=True)
        for name, (fn, max_hold) in strategies.items():
            entries = fn(df)
            trades = simulate_entries(df, period, name, entries, max_hold_bars=max_hold)
            trade_dicts = []
            for t in trades:
                row = dataclasses.asdict(t)
                row["entry_time"] = t.entry_time.isoformat()
                row["exit_time"] = t.exit_time.isoformat()
                trade_dicts.append(row)
                all_rows.append(row)
            m = metrics(trades)
            summary_json["strategies"][f"{period}:{name}"] = {k: v for k, v in m.items() if k != "equity_curve"}
            summary_rows.append({"period": period, "strategy": name, **{k: v for k, v in m.items() if k != "equity_curve"}})
            for c in m["equity_curve"]:
                curve_rows.append({"period": period, "strategy": name, **c})
            write_csv(out_dir / f"{period}_{name}_trades.csv", trade_dicts)
    write_csv(out_dir / "all_trades.csv", all_rows)
    write_csv(out_dir / "metrics_comparison.csv", summary_rows)
    write_csv(out_dir / "equity_curves.csv", curve_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")

    ranked = sorted(summary_rows, key=lambda r: (float(r["profit_factor"]), float(r["return_pct"]), int(r["trades"])), reverse=True)
    md = [
        "# Twitter XAUUSD Candidate Backtest",
        "",
        f"Generated: {summary_json['generated_at']}",
        f"Data: {summary_json['input_start']} → {summary_json['input_end']} M5",
        "",
        "## Assumptions",
        f"- Initial equity: {INITIAL_EQUITY}",
        f"- Risk per trade: {RISK_PCT * 100:.2f}%",
        f"- Commission: ${COMMISSION_PER_LOT_ROUND}/lot round-turn",
        f"- Spread cost: {SPREAD_PRICE_COST:.2f} price units per round-trip",
        "- Entry at next bar open; if SL and TP both hit in one M5 bar, SL is assumed first.",
        "",
        "## Ranking",
    ]
    for row in ranked:
        md.append(
            f"- {row['period']} {row['strategy']}: trades={row['trades']}, "
            f"return={row['return_pct']}%, PF={row['profit_factor']}, "
            f"DD={row['max_dd_pct']}%, exp=${row['expectancy_usd']}, win={row['win_rate_pct']}%"
        )
    md += [
        "",
        "## Strategy mapping",
        "- london_asian_range_breakout: Twitter London Open + Asian range breakout idea.",
        "- pdh_pdl_sweep_reversal: PDH/PDL stop-hunt / liquidity sweep reversal idea.",
        "- us_session_orb: GC/XAUUSD US session ORB idea.",
        "- range_false_breakout_reversal: range compression false-breakout reversal idea.",
        "- ema20_pullback_trend: sell-rallies / buy-dips at MA support/resistance idea.",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "ranked": ranked}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
