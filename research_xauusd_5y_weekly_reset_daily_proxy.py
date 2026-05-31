#!/usr/bin/env python3
"""5-year weekly-reset XAUUSD/Gold daily-bar proxy backtest.

Yahoo Finance only exposes 5y daily data for GC=F/XAUUSD-like gold series; intraday is
limited to ~2 years. This script keeps the same weekly-reset sampling framework but
runs a lower-frequency proxy of the trend strategy on daily OHLC bars.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yfinance as yf

OUT_ROOT = Path("/home/chain4655/Documents/backtest_reports/xauusd_5y_weekly_reset")
CONTRACT_SIZE = 100.0


@dataclasses.dataclass(frozen=True)
class Config:
    start_equity: float = 100000.0
    risk_pct: float = 0.0075
    daily_dd_limit: float = 0.03
    total_dd_limit: float = 0.045
    profit_target: float = 0.10
    max_lots: float = 1.2
    fast_sma: int = 20
    slow_sma: int = 60
    atr_period: int = 14
    breakout_lookback: int = 20
    trend_threshold: float = 0.35
    stop_atr: float = 2.5
    reward_multiple: float = 1.2
    cost_per_lot_roundtrip: float = 40.0


@dataclasses.dataclass
class Position:
    direction: str
    entry_time: pd.Timestamp
    entry: float
    sl: float
    tp: float
    volume: float


def flatten_yf(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(c[0]).lower().replace(" ", "_") for c in out.columns]
    else:
        out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    return out.rename(columns={"date": "time", "datetime": "time"})


def fetch_daily(ticker: str = "GC=F") -> pd.DataFrame:
    df = yf.download(ticker, period="5y", interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"Yahoo download returned empty data for {ticker}")
    out = flatten_yf(df)
    out = out[["time", "open", "high", "low", "close", "volume"]].dropna()
    out["time"] = pd.to_datetime(out["time"]).dt.tz_localize(None)
    return out.sort_values("time").reset_index(drop=True)


def add_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(cfg.atr_period).mean()
    out["fast_sma"] = close.rolling(cfg.fast_sma).mean()
    out["slow_sma"] = close.rolling(cfg.slow_sma).mean()
    out["momentum"] = ((close - close.shift(3)) / out["atr"]).clip(-2, 2)
    out["recent_high"] = high.rolling(cfg.breakout_lookback).max()
    out["recent_low"] = low.rolling(cfg.breakout_lookback).min()
    trend = pd.Series(0.0, index=out.index)
    trend[(close > out["fast_sma"]) & (out["fast_sma"] > out["slow_sma"])] = 0.55
    trend[(close < out["fast_sma"]) & (out["fast_sma"] < out["slow_sma"])] = -0.55
    trend[(trend == 0.0) & (close > out["slow_sma"])] = 0.20
    trend[(trend == 0.0) & (close < out["slow_sma"])] = -0.20
    breakout = (((close - out["recent_high"]) / out["atr"]).clip(-1, 1) * 0.30) + (((out["recent_low"] - close) / out["atr"]).clip(-1, 1) * -0.30)
    out["score"] = (trend + breakout + out["momentum"].clip(-1.5, 1.5) * 0.25).clip(-1.5, 1.5)
    out["signal"] = "NONE"
    out.loc[out["score"] >= cfg.trend_threshold, "signal"] = "BUY"
    out.loc[out["score"] <= -cfg.trend_threshold, "signal"] = "SELL"
    return out.dropna().reset_index(drop=True)


def pnl(direction: str, entry: float, exit_price: float, volume: float, cfg: Config) -> float:
    gross = (exit_price - entry) * CONTRACT_SIZE * volume if direction == "BUY" else (entry - exit_price) * CONTRACT_SIZE * volume
    return gross - cfg.cost_per_lot_roundtrip * volume


def exit_hit(pos: Position, row: pd.Series) -> Tuple[Optional[str], Optional[float]]:
    high = float(row["high"])
    low = float(row["low"])
    if pos.direction == "BUY":
        if low <= pos.sl:
            return "SL", pos.sl
        if high >= pos.tp:
            return "TP", pos.tp
    else:
        if high >= pos.sl:
            return "SL", pos.sl
        if low <= pos.tp:
            return "TP", pos.tp
    return None, None


def volume_for(equity: float, entry: float, sl: float, cfg: Config) -> float:
    risk_per_lot = abs(entry - sl) * CONTRACT_SIZE
    vol = min(cfg.max_lots, (equity * cfg.risk_pct) / max(risk_per_lot, 1e-9))
    return max(0.01, math.floor(vol / 0.01) * 0.01)


def sl_tp(direction: str, entry: float, atr: float, cfg: Config) -> Tuple[float, float]:
    dist = atr * cfg.stop_atr
    if direction == "BUY":
        return entry - dist, entry + dist * cfg.reward_multiple
    return entry + dist, entry - dist * cfg.reward_multiple


def simulate_week(week_df: pd.DataFrame, cfg: Config) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    balance = cfg.start_equity
    max_equity = cfg.start_equity
    pos: Optional[Position] = None
    trades: List[Dict[str, object]] = []
    stopped = ""

    def close_pos(row: pd.Series, price: float, reason: str) -> None:
        nonlocal balance, pos
        if pos is None:
            return
        gain = pnl(pos.direction, pos.entry, price, pos.volume, cfg)
        balance += gain
        trades.append({
            "week": row["week"], "entry_time": pos.entry_time.isoformat(), "exit_time": pd.Timestamp(row["time"]).isoformat(),
            "direction": pos.direction, "entry": round(pos.entry, 2), "exit": round(price, 2),
            "volume": round(pos.volume, 2), "pnl": round(gain, 2), "reason": reason,
        })
        pos = None

    for _, row in week_df.iterrows():
        t = pd.Timestamp(row["time"])
        close = float(row["close"])
        equity = balance + (pnl(pos.direction, pos.entry, close, pos.volume, dataclasses.replace(cfg, cost_per_lot_roundtrip=0.0)) if pos else 0.0)
        max_equity = max(max_equity, equity)
        if 1.0 - equity / max_equity >= cfg.total_dd_limit:
            if pos:
                close_pos(row, close, "TOTAL_DD")
            stopped = "total_dd"
            break
        if equity / cfg.start_equity - 1.0 >= cfg.profit_target:
            if pos:
                close_pos(row, close, "PROFIT_TARGET")
            stopped = "profit_target"
            break

        if pos:
            reason, price = exit_hit(pos, row)
            if reason and price is not None:
                close_pos(row, float(price), reason)
            elif (pos.direction == "BUY" and row["signal"] == "SELL") or (pos.direction == "SELL" and row["signal"] == "BUY"):
                close_pos(row, close, "REVERSE")

        if pos is None and row["signal"] in {"BUY", "SELL"}:
            direction = str(row["signal"])
            sl, tp = sl_tp(direction, close, float(row["atr"]), cfg)
            vol = volume_for(balance, close, sl, cfg)
            pos = Position(direction, t, close, sl, tp, vol)

    if pos and not week_df.empty:
        close_pos(week_df.iloc[-1], float(week_df.iloc[-1]["close"]), "WEEK_END")

    ret = balance / cfg.start_equity - 1.0
    return {
        "week": str(week_df["week"].iloc[0]),
        "start": pd.Timestamp(week_df["time"].iloc[0]).isoformat(),
        "end": pd.Timestamp(week_df["time"].iloc[-1]).isoformat(),
        "bars": int(len(week_df)),
        "return_pct": round(ret * 100, 4),
        "pnl": round(balance - cfg.start_equity, 2),
        "end_equity": round(balance, 2),
        "trades": len(trades),
        "wins": sum(1 for t in trades if float(t["pnl"]) > 0),
        "losses": sum(1 for t in trades if float(t["pnl"]) < 0),
        "stopped_reason": stopped,
    }, trades


def summarize(rows: Sequence[Dict[str, object]], trades: Sequence[Dict[str, object]], cfg: Config, df: pd.DataFrame) -> Dict[str, object]:
    rets = [float(r["return_pct"]) for r in rows]
    wins = [x for x in rets if x > 0]
    std = statistics.pstdev(rets) if len(rets) > 1 else 0.0
    avg = statistics.fmean(rets) if rets else 0.0
    gross_win = sum(float(t["pnl"]) for t in trades if float(t["pnl"]) > 0)
    gross_loss = -sum(float(t["pnl"]) for t in trades if float(t["pnl"]) < 0)
    return {
        "variant": "yahoo_gc_f_daily_5y_proxy_weekly_reset",
        "data_source": "Yahoo Finance GC=F daily; proxy for XAUUSD spot",
        "data_start": pd.Timestamp(df["time"].min()).isoformat(),
        "data_end": pd.Timestamp(df["time"].max()).isoformat(),
        "weeks": len(rows),
        "active_weeks": sum(1 for r in rows if int(r["trades"]) > 0),
        "weekly_win_rate_pct": round(len(wins) / len(rets) * 100, 2) if rets else 0.0,
        "avg_week_return_pct": round(avg, 4),
        "median_week_return_pct": round(statistics.median(rets), 4) if rets else 0.0,
        "std_week_return_pct": round(std, 4),
        "sharpe_weekly_annualized": round((avg / std) * math.sqrt(52), 3) if std else 0.0,
        "best_week_pct": round(max(rets), 4) if rets else 0.0,
        "worst_week_pct": round(min(rets), 4) if rets else 0.0,
        "sum_independent_week_pnl": round(sum(float(r["pnl"]) for r in rows), 2),
        "trades": len(trades),
        "avg_trades_per_week": round(len(trades) / len(rows), 2) if rows else 0.0,
        "trade_win_rate_pct": round(sum(1 for t in trades if float(t["pnl"]) > 0) / len(trades) * 100, 2) if trades else 0.0,
        "trade_profit_factor": round(gross_win / gross_loss, 3) if gross_loss else 999.0,
        "total_dd_stops": sum(1 for r in rows if r["stopped_reason"] == "total_dd"),
        "profit_target_stops": sum(1 for r in rows if r["stopped_reason"] == "profit_target"),
        "config": dataclasses.asdict(cfg),
    }


def main() -> int:
    cfg = Config()
    raw = fetch_daily()
    df = add_indicators(raw, cfg)
    iso = df["time"].dt.isocalendar()
    df["week"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    out_dir = OUT_ROOT / ("daily_proxy_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out_dir / "input_gc_f_daily_raw.csv", index=False)
    df.to_csv(out_dir / "input_gc_f_daily_with_indicators.csv", index=False)
    rows: List[Dict[str, object]] = []
    trades: List[Dict[str, object]] = []
    for _, week_df in df.groupby("week", sort=True):
        row, trs = simulate_week(week_df.reset_index(drop=True), cfg)
        rows.append(row)
        trades.extend(trs)
    pd.DataFrame(rows).to_csv(out_dir / "daily_proxy_weekly_returns.csv", index=False)
    pd.DataFrame(trades).to_csv(out_dir / "daily_proxy_trades.csv", index=False)
    summary = summarize(rows, trades, cfg, df)
    (out_dir / "daily_proxy_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
