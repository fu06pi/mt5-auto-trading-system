#!/usr/bin/env python3
"""Research-only XAUUSD rebound/reversion backtest.

Tests independent counter-trend BUY sleeves for bearish HTF conditions:
1) RSI bullish-divergence rebound
2) Linear-regression residual rebound

No MT5 orders and no active_plan edits.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path("/home/chain4655/Documents/Projects/MT5")
DEFAULT_DATA = Path("/home/chain4655/Documents/backtest_reports/xauusd_5y_weekly_reset/data/XAUUSD_M5_5y_mt5.csv")
OUT_DIR = ROOT / "backtest_reports_xauusd_rebound_reversion"
INITIAL_EQUITY = 100000.0
CONTRACT_SIZE = 100.0
POINT = 0.01
SPREAD_POINTS = 25.0
COMMISSION_PER_LOT = 7.0


@dataclass(frozen=True)
class Params:
    name: str
    mode: str
    rsi_period: int = 14
    atr_period: int = 14
    fast_sma: int = 20
    mid_sma: int = 50
    htf_fast: int = 50
    htf_slow: int = 200
    div_lookback: int = 36
    div_min_rsi_delta: float = 4.0
    div_low_tolerance_atr: float = 0.25
    reg_window: int = 48
    reg_z_entry: float = -1.35
    reclaim_sma: bool = True
    min_rebound_mom_atr: float = 0.25
    max_adx: float = 38.0
    min_atr_ratio: float = 0.55
    max_atr_ratio: float = 2.20
    risk_pct: float = 0.001
    max_lots: float = 0.35
    stop_atr: float = 1.15
    reward_multiple: float = 1.25
    tp_mode: str = "sma50"  # sma50 or fixed_rr
    max_hold_bars: int = 24
    cooldown_bars: int = 8
    sessions: str = "all"  # all, asia_london, us_london_overlap


@dataclass(frozen=True)
class Trade:
    strategy: str
    entry_time: str
    exit_time: str
    side: str
    entry: float
    exit: float
    sl: float
    tp: float
    volume: float
    net_pnl: float
    gross_pnl: float
    commission: float
    spread_cost: float
    r_multiple: float
    reason: str
    bars_held: int
    session: str
    rsi: float
    adx: float
    atr: float
    reg_z: float
    htf: str


def session_of(ts: pd.Timestamp) -> str:
    hour = int(ts.hour)
    if 0 <= hour < 7:
        return "asia"
    if 7 <= hour < 13:
        return "london_pre_us"
    if 13 <= hour < 20:
        return "us_london_overlap"
    return "late_us"


def load_bars(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"Time": "time", "Open": "open", "High": "high", "Low": "low", "Close": "close"})
    required = ["time", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates("time")
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def add_indicators(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(p.atr_period).mean()
    out["atr_long"] = tr.rolling(288).mean()
    out["sma20"] = close.rolling(p.fast_sma).mean()
    out["sma50"] = close.rolling(p.mid_sma).mean()
    out["htf_fast"] = close.rolling(p.htf_fast * 12).mean()
    out["htf_slow"] = close.rolling(p.htf_slow * 12).mean()
    out["htf"] = "NEUTRAL"
    out.loc[out["htf_fast"] > out["htf_slow"], "htf"] = "BULL"
    out.loc[out["htf_fast"] < out["htf_slow"], "htf"] = "BEAR"

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(p.rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(p.rsi_period).mean()
    rs = gain / loss.mask(loss == 0, math.nan)
    out["rsi"] = (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr_sum = tr.rolling(14).sum().mask(lambda x: x == 0, math.nan)
    plus_di = 100.0 * plus_dm.rolling(14).sum() / tr_sum
    minus_di = 100.0 * minus_dm.rolling(14).sum() / tr_sum
    denom = (plus_di + minus_di).mask((plus_di + minus_di) == 0, math.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom
    out["adx"] = dx.rolling(14).mean().fillna(50.0)

    # Fast rolling linear-regression residual z-score with fixed x=0..window-1.
    # For each window, sum_xy is computed by a small raw rolling apply; the rest is vectorized.
    x_vals = list(range(p.reg_window))
    x_sum = float(sum(x_vals))
    x2_sum = float(sum(v * v for v in x_vals))
    denom = p.reg_window * x2_sum - x_sum * x_sum

    def sum_xy(vals: Sequence[float]) -> float:
        return float(sum(float(v) * x for x, v in zip(x_vals, vals)))

    y_sum = close.rolling(p.reg_window).sum()
    y2_sum = (close * close).rolling(p.reg_window).sum()
    xy_sum = close.rolling(p.reg_window).apply(sum_xy, raw=True)
    slope = (p.reg_window * xy_sum - x_sum * y_sum) / denom
    intercept = (y_sum - slope * x_sum) / p.reg_window
    fitted_last = intercept + slope * (p.reg_window - 1)
    resid_last = close - fitted_last
    sse = y2_sum - 2 * intercept * y_sum - 2 * slope * xy_sum + p.reg_window * intercept * intercept + 2 * intercept * slope * x_sum + slope * slope * x2_sum
    resid_std = (sse.clip(lower=0) / p.reg_window).pow(0.5)
    out["reg_z"] = (resid_last / resid_std.mask(resid_std == 0, math.nan)).fillna(0.0)
    out["mom_atr"] = (close - close.shift(3)) / out["atr"]
    out["atr_ratio"] = out["atr"] / out["atr_long"]
    out["session"] = out["time"].map(session_of)
    return out


def session_allowed(session: str, p: Params) -> bool:
    if p.sessions == "all":
        return True
    if p.sessions == "asia_london":
        return session in {"asia", "london_pre_us"}
    if p.sessions == "us_london_overlap":
        return session == "us_london_overlap"
    return True


def bullish_divergence(data: pd.DataFrame, i: int, p: Params) -> bool:
    row = data.iloc[i]
    if i < p.div_lookback + 2 or row["atr"] <= 0:
        return False
    window = data.iloc[i - p.div_lookback:i]
    swing_idx = int(window["low"].idxmin())
    swing = data.loc[swing_idx]
    low_ok = row["low"] <= float(swing["low"]) + p.div_low_tolerance_atr * float(row["atr"])
    rsi_ok = float(row["rsi"]) >= float(swing["rsi"]) + p.div_min_rsi_delta
    reclaim_ok = (not p.reclaim_sma) or float(row["close"]) > float(row["sma20"])
    return bool(low_ok and rsi_ok and reclaim_ok)


def should_enter(data: pd.DataFrame, i: int, p: Params) -> bool:
    row = data.iloc[i]
    prev = data.iloc[i - 1]
    if row["htf"] != "BEAR":
        return False
    if not session_allowed(str(row["session"]), p):
        return False
    if row["adx"] > p.max_adx:
        return False
    if not (p.min_atr_ratio <= row["atr_ratio"] <= p.max_atr_ratio):
        return False
    if row["mom_atr"] < p.min_rebound_mom_atr:
        return False
    if p.mode == "divergence":
        return bullish_divergence(data, i, p)
    if p.mode == "regression":
        was_stretched = float(prev["reg_z"]) <= p.reg_z_entry
        reclaim = float(row["close"]) > float(row["sma20"]) if p.reclaim_sma else float(row["reg_z"]) > float(prev["reg_z"])
        return bool(was_stretched and reclaim)
    return False


def calc_lots(equity: float, risk_pct: float, risk_distance: float, max_lots: float) -> float:
    if risk_distance <= 0:
        return 0.0
    lots = equity * risk_pct / (risk_distance * CONTRACT_SIZE)
    lots = min(max_lots, math.floor(lots * 100) / 100.0)
    return max(0.01, lots) if lots >= 0.01 else 0.0


def close_trade(pos: Dict[str, Any], row: pd.Series, exit_price: float, reason: str, bars_held: int) -> Trade:
    gross = (exit_price - pos["entry"]) * pos["volume"] * CONTRACT_SIZE
    commission = COMMISSION_PER_LOT * pos["volume"]
    spread_cost = SPREAD_POINTS * POINT * CONTRACT_SIZE * pos["volume"]
    net = gross - commission - spread_cost
    risk_usd = abs(pos["entry"] - pos["sl"]) * pos["volume"] * CONTRACT_SIZE
    return Trade(
        strategy=pos["strategy"],
        entry_time=pos["entry_time"].isoformat(),
        exit_time=row["time"].isoformat(),
        side="BUY",
        entry=round(pos["entry"], 2),
        exit=round(float(exit_price), 2),
        sl=round(pos["sl"], 2),
        tp=round(pos["tp"], 2),
        volume=round(pos["volume"], 2),
        net_pnl=round(net, 2),
        gross_pnl=round(gross, 2),
        commission=round(commission, 2),
        spread_cost=round(spread_cost, 2),
        r_multiple=round(net / max(risk_usd, 1e-9), 3),
        reason=reason,
        bars_held=bars_held,
        session=pos["session"],
        rsi=round(pos["rsi"], 2),
        adx=round(pos["adx"], 2),
        atr=round(pos["atr"], 2),
        reg_z=round(pos["reg_z"], 2),
        htf=pos["htf"],
    )


def backtest(df: pd.DataFrame, p: Params) -> Tuple[List[Trade], Dict[str, Any]]:
    data = add_indicators(df, p).dropna().reset_index(drop=True)
    trades: List[Trade] = []
    equity = INITIAL_EQUITY
    peak = equity
    max_dd = 0.0
    pos: Optional[Dict[str, Any]] = None
    cooldown_until = -1
    for i in range(1, len(data)):
        row = data.iloc[i]
        if pos is not None:
            bars_held = i - int(pos["entry_idx"])
            exit_price = None
            reason = ""
            if row["low"] <= pos["sl"]:
                exit_price, reason = pos["sl"], "SL"
            elif row["high"] >= pos["tp"]:
                exit_price, reason = pos["tp"], "TP"
            elif bars_held >= p.max_hold_bars:
                exit_price, reason = float(row["close"]), "TIME"
            if exit_price is not None:
                trade = close_trade(pos, row, float(exit_price), reason, bars_held)
                trades.append(trade)
                equity += trade.net_pnl
                peak = max(peak, equity)
                max_dd = max(max_dd, (peak - equity) / max(peak, 1.0))
                pos = None
                cooldown_until = i + p.cooldown_bars
        if pos is None and i >= cooldown_until and should_enter(data, i, p):
            entry = float(row["close"])
            sl = min(float(row["low"] - 0.35 * row["atr"]), entry - p.stop_atr * float(row["atr"]))
            tp = float(row["sma50"]) if p.tp_mode == "sma50" else entry + (entry - sl) * p.reward_multiple
            if tp <= entry:
                tp = entry + (entry - sl) * p.reward_multiple
            volume = calc_lots(equity, p.risk_pct, abs(entry - sl), p.max_lots)
            if volume <= 0:
                continue
            pos = {
                "strategy": p.name,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "volume": volume,
                "entry_idx": i,
                "entry_time": row["time"],
                "session": row["session"],
                "rsi": row["rsi"],
                "adx": row["adx"],
                "atr": row["atr"],
                "reg_z": row["reg_z"],
                "htf": row["htf"],
            }
    return trades, summarize(trades, equity, max_dd, data)


def summarize(trades: Sequence[Trade], final_equity: float, max_dd: float, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]
    gp = sum(t.net_pnl for t in wins)
    gl = -sum(t.net_pnl for t in losses)
    out = {
        "trades": len(trades),
        "net_pnl": round(final_equity - INITIAL_EQUITY, 2),
        "return_pct": round((final_equity / INITIAL_EQUITY - 1.0) * 100.0, 2),
        "win_rate_pct": round(100.0 * len(wins) / len(trades), 2) if trades else 0.0,
        "profit_factor": round(gp / gl, 3) if gl > 0 else (999.0 if gp > 0 else 0.0),
        "max_dd_pct": round(max_dd * 100.0, 2),
        "expectancy_usd": round(sum(t.net_pnl for t in trades) / len(trades), 2) if trades else 0.0,
        "avg_r": round(sum(t.r_multiple for t in trades) / len(trades), 3) if trades else 0.0,
        "total_costs": round(sum(t.commission + t.spread_cost for t in trades), 2),
        "tp": sum(1 for t in trades if t.reason == "TP"),
        "sl": sum(1 for t in trades if t.reason == "SL"),
        "time_exit": sum(1 for t in trades if t.reason == "TIME"),
    }
    if data is not None and len(data):
        out.update({"data_bars": len(data), "data_start": data.iloc[0]["time"].isoformat(), "data_end": data.iloc[-1]["time"].isoformat()})
    return out


def grouped(trades: Sequence[Trade], key: str) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Trade]] = {}
    for t in trades:
        groups.setdefault(str(getattr(t, key)), []).append(t)
    return {k: summarize(v, INITIAL_EQUITY + sum(t.net_pnl for t in v), 0.0) for k, v in sorted(groups.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    df = load_bars(args.data)
    variants = [
        Params(name="div_sma50_all", mode="divergence", tp_mode="sma50"),
        Params(name="div_1p25r_all", mode="divergence", tp_mode="fixed_rr", reward_multiple=1.25),
        Params(name="div_sma50_asia_london", mode="divergence", tp_mode="sma50", sessions="asia_london"),
        Params(name="reg_sma50_all", mode="regression", tp_mode="sma50"),
        Params(name="reg_1p25r_all", mode="regression", tp_mode="fixed_rr", reward_multiple=1.25),
        Params(name="reg_sma50_overlap", mode="regression", tp_mode="sma50", sessions="us_london_overlap"),
        Params(name="reg_loose_1p1r", mode="regression", tp_mode="fixed_rr", reward_multiple=1.1, reg_z_entry=-1.0, max_adx=45.0),
        Params(name="div_loose_1p1r", mode="divergence", tp_mode="fixed_rr", reward_multiple=1.1, div_min_rsi_delta=2.0, max_adx=45.0),
    ]
    summary: List[Dict[str, Any]] = []
    all_trades: List[Trade] = []
    diagnostics: Dict[str, Any] = {
        "data_path": str(args.data),
        "raw_bars": len(df),
        "raw_start": df.iloc[0]["time"].isoformat(),
        "raw_end": df.iloc[-1]["time"].isoformat(),
        "assumptions": {
            "symbol": "XAUUSD",
            "timeframe": "M5",
            "initial_equity": INITIAL_EQUITY,
            "contract_size": CONTRACT_SIZE,
            "spread_points": SPREAD_POINTS,
            "commission_per_lot": COMMISSION_PER_LOT,
            "side": "BUY only, counter-trend rebound while HTF is BEAR",
            "intrabar_order": "conservative: SL before TP if both touched",
            "live_side_effects": "none; script is research-only and reads CSV cache",
        },
        "grouped": {},
    }
    for p in variants:
        trades, metrics = backtest(df, p)
        summary.append({"strategy": p.name, **metrics})
        all_trades.extend(trades)
        diagnostics["grouped"][p.name] = {"session": grouped(trades, "session")}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(OUT_DIR / "xauusd_rebound_reversion_summary.csv", index=False)
    pd.DataFrame([asdict(t) for t in all_trades]).to_csv(OUT_DIR / "xauusd_rebound_reversion_trades.csv", index=False)
    with (OUT_DIR / "xauusd_rebound_reversion_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)
    active = ROOT / "auto_quant/active_plan.json"
    print(json.dumps({"summary": summary, "out_dir": str(OUT_DIR), "active_plan_mtime": active.stat().st_mtime, "active_plan_size": active.stat().st_size}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
