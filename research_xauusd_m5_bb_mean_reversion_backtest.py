#!/usr/bin/env python3
"""Research-only XAUUSD M5 Bollinger mean-reversion backtest.

No MT5 orders, no active_plan edits. Uses cached broker M5 CSV by default.
"""
from __future__ import annotations

# pyright: reportAttributeAccessIssue=false, reportAssignmentType=false

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path("/home/chain4655/Documents/Projects/MT5")
DEFAULT_DATA = Path("/home/chain4655/Documents/backtest_reports/xauusd_5y_weekly_reset/data/XAUUSD_M5_5y_mt5.csv")
OUT_DIR = ROOT / "backtest_reports_bb_m5_mean_reversion"
INITIAL_EQUITY = 100000.0
CONTRACT_SIZE = 100.0
SPREAD_POINTS = 20.0
POINT = 0.01
COMMISSION_PER_LOT = 7.0


@dataclass(frozen=True)
class Params:
    name: str
    bb_period: int = 20
    bb_dev: float = 2.0
    rsi_period: int = 14
    rsi_buy: float = 35.0
    rsi_sell: float = 65.0
    adx_max: float = 20.0
    bb_width_mult_max: float = 1.20
    atr_expansion_max: float = 1.35
    atr_sl: float = 1.20
    tp_mode: str = "middle"  # middle or fixed_rr
    reward_multiple: float = 1.50
    max_hold_bars: int = 18
    cooldown_bars: int = 6
    risk_pct: float = 0.001
    max_lots: float = 0.5
    wick_body_min: float = 1.2
    sessions: str = "all"  # all, asia, asia_london


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
    gross_pnl: float
    net_pnl: float
    commission: float
    spread_cost: float
    r_multiple: float
    reason: str
    bars_held: int
    session: str
    rsi: float
    adx: float
    bb_width: float
    atr: float


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
    rename = {"Time": "time", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "tick_volume"}
    df = df.rename(columns=rename)
    required = ["time", "open", "high", "low", "close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if "tick_volume" not in df.columns:
        df["tick_volume"] = 0.0
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates("time")
    for col in ["open", "high", "low", "close", "tick_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return df


def add_indicators(df: pd.DataFrame, params: Params) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    prev_close = close.shift(1)

    out["bb_mid"] = close.rolling(params.bb_period).mean()
    out["bb_std"] = close.rolling(params.bb_period).std(ddof=0)
    out["bb_upper"] = out["bb_mid"] + params.bb_dev * out["bb_std"]
    out["bb_lower"] = out["bb_mid"] - params.bb_dev * out["bb_std"]
    out["bb_width"] = out["bb_upper"] - out["bb_lower"]
    out["bb_width_avg"] = out["bb_width"].rolling(96).mean()

    tr: pd.Series = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    out["atr"] = tr.rolling(14).mean()
    out["atr_long"] = tr.rolling(96).mean()

    delta: pd.Series = close.diff()
    gain: pd.Series = delta.clip(lower=0).rolling(params.rsi_period).mean()
    loss: pd.Series = (-delta.clip(upper=0)).rolling(params.rsi_period).mean()
    safe_loss: pd.Series = loss.mask(loss == 0, math.nan)
    rs: pd.Series = gain / safe_loss
    out["rsi"] = 100.0 - (100.0 / (1.0 + rs))
    out["rsi"] = out["rsi"].fillna(50.0)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr_sum = tr.rolling(14).sum()
    safe_tr_sum = tr_sum.mask(tr_sum == 0, math.nan)
    plus_di = 100.0 * plus_dm.rolling(14).sum() / safe_tr_sum
    minus_di = 100.0 * minus_dm.rolling(14).sum() / safe_tr_sum
    denom = (plus_di + minus_di).mask((plus_di + minus_di) == 0, math.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom
    out["adx"] = dx.rolling(14).mean().fillna(50.0)

    body = (out["close"] - out["open"]).abs().clip(lower=POINT)
    out["upper_wick_ratio"] = (out["high"] - out[["open", "close"]].max(axis=1)) / body
    out["lower_wick_ratio"] = (out[["open", "close"]].min(axis=1) - out["low"]) / body
    out["session"] = out["time"].map(session_of)
    return out


def session_allowed(session: str, params: Params) -> bool:
    if params.sessions == "all":
        return True
    if params.sessions == "asia":
        return session == "asia"
    if params.sessions == "asia_london":
        return session in {"asia", "london_pre_us"}
    return True


def calc_lots(equity: float, risk_pct: float, risk_distance: float, max_lots: float) -> float:
    if risk_distance <= 0:
        return 0.0
    lots = equity * risk_pct / (risk_distance * CONTRACT_SIZE)
    lots = min(max_lots, math.floor(lots * 100) / 100.0)
    return max(0.01, lots) if lots >= 0.01 else 0.0


def should_enter(row: pd.Series, prev: pd.Series, params: Params) -> Optional[str]:
    if not session_allowed(str(row["session"]), params):
        return None
    if row["adx"] > params.adx_max:
        return None
    if row["bb_width"] > row["bb_width_avg"] * params.bb_width_mult_max:
        return None
    if row["atr"] > row["atr_long"] * params.atr_expansion_max:
        return None

    buy = (
        row["low"] < row["bb_lower"]
        and row["close"] > row["bb_lower"]
        and row["rsi"] <= params.rsi_buy
        and row["rsi"] >= prev["rsi"]
        and row["lower_wick_ratio"] >= params.wick_body_min
    )
    sell = (
        row["high"] > row["bb_upper"]
        and row["close"] < row["bb_upper"]
        and row["rsi"] >= params.rsi_sell
        and row["rsi"] <= prev["rsi"]
        and row["upper_wick_ratio"] >= params.wick_body_min
    )
    if buy and not sell:
        return "BUY"
    if sell and not buy:
        return "SELL"
    return None


def backtest(df: pd.DataFrame, params: Params) -> Tuple[List[Trade], Dict[str, Any]]:
    data = add_indicators(df, params).dropna().reset_index(drop=True)
    trades: List[Trade] = []
    equity = INITIAL_EQUITY
    peak = equity
    max_dd = 0.0
    pos: Optional[Dict[str, Any]] = None
    cooldown_until = -1

    for i in range(1, len(data)):
        row = data.iloc[i]
        prev = data.iloc[i - 1]

        if pos is not None:
            bars_held = i - int(pos["entry_idx"])
            exit_price = None
            reason = ""
            if pos["side"] == "BUY":
                if row["low"] <= pos["sl"]:
                    exit_price, reason = pos["sl"], "SL"
                elif row["high"] >= pos["tp"]:
                    exit_price, reason = pos["tp"], "TP"
            else:
                if row["high"] >= pos["sl"]:
                    exit_price, reason = pos["sl"], "SL"
                elif row["low"] <= pos["tp"]:
                    exit_price, reason = pos["tp"], "TP"
            if exit_price is None and bars_held >= params.max_hold_bars:
                exit_price, reason = float(row["close"]), "TIME"
            if exit_price is not None:
                sign = 1.0 if pos["side"] == "BUY" else -1.0
                gross = (exit_price - pos["entry"]) * sign * pos["volume"] * CONTRACT_SIZE
                commission = COMMISSION_PER_LOT * pos["volume"]
                spread_cost = SPREAD_POINTS * POINT * CONTRACT_SIZE * pos["volume"]
                net = gross - commission - spread_cost
                equity += net
                peak = max(peak, equity)
                max_dd = max(max_dd, (peak - equity) / max(peak, 1.0))
                risk_usd = abs(pos["entry"] - pos["sl"]) * pos["volume"] * CONTRACT_SIZE
                trades.append(Trade(
                    strategy=params.name,
                    entry_time=pos["entry_time"].isoformat(),
                    exit_time=row["time"].isoformat(),
                    side=pos["side"],
                    entry=round(pos["entry"], 2),
                    exit=round(float(exit_price), 2),
                    sl=round(pos["sl"], 2),
                    tp=round(pos["tp"], 2),
                    volume=round(pos["volume"], 2),
                    gross_pnl=round(gross, 2),
                    net_pnl=round(net, 2),
                    commission=round(commission, 2),
                    spread_cost=round(spread_cost, 2),
                    r_multiple=round(net / max(risk_usd, 1e-9), 3),
                    reason=reason,
                    bars_held=bars_held,
                    session=pos["session"],
                    rsi=round(pos["rsi"], 2),
                    adx=round(pos["adx"], 2),
                    bb_width=round(pos["bb_width"], 2),
                    atr=round(pos["atr"], 2),
                ))
                pos = None
                cooldown_until = i + params.cooldown_bars

        if pos is None and i >= cooldown_until:
            side = should_enter(row, prev, params)
            if side is None:
                continue
            entry = float(row["close"])
            if side == "BUY":
                sl = min(float(row["low"] - 0.5 * row["atr"]), entry - params.atr_sl * float(row["atr"]))
                tp = float(row["bb_mid"]) if params.tp_mode == "middle" else entry + (entry - sl) * params.reward_multiple
            else:
                sl = max(float(row["high"] + 0.5 * row["atr"]), entry + params.atr_sl * float(row["atr"]))
                tp = float(row["bb_mid"]) if params.tp_mode == "middle" else entry - (sl - entry) * params.reward_multiple
            risk_distance = abs(entry - sl)
            volume = calc_lots(equity, params.risk_pct, risk_distance, params.max_lots)
            if volume <= 0:
                continue
            pos = {
                "side": side,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "volume": volume,
                "entry_idx": i,
                "entry_time": row["time"],
                "session": row["session"],
                "rsi": row["rsi"],
                "adx": row["adx"],
                "bb_width": row["bb_width"],
                "atr": row["atr"],
            }

    metrics = summarize(trades, equity, max_dd)
    metrics.update({"data_bars": len(data), "data_start": data.iloc[0]["time"].isoformat(), "data_end": data.iloc[-1]["time"].isoformat()})
    return trades, metrics


def summarize(trades: Sequence[Trade], final_equity: float, max_dd: float) -> Dict[str, Any]:
    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]
    gp = sum(t.net_pnl for t in wins)
    gl = -sum(t.net_pnl for t in losses)
    return {
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


def grouped(rows: Sequence[Trade], key: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, List[Trade]] = {}
    for trade in rows:
        out.setdefault(str(getattr(trade, key)), []).append(trade)
    return {k: summarize(v, INITIAL_EQUITY + sum(t.net_pnl for t in v), 0.0) for k, v in sorted(out.items())}


def write_outputs(summary: List[Dict[str, Any]], trades: List[Trade], diagnostics: Dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(OUT_DIR / "bb_m5_mean_reversion_summary.csv", index=False)
    pd.DataFrame([asdict(t) for t in trades]).to_csv(OUT_DIR / "bb_m5_mean_reversion_trades.csv", index=False)
    with (OUT_DIR / "bb_m5_mean_reversion_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()

    df = load_bars(args.data)
    variants = [
        Params(name="bb_m5_middle_all"),
        Params(name="bb_m5_middle_asia", sessions="asia"),
        Params(name="bb_m5_middle_asia_london", sessions="asia_london"),
        Params(name="bb_m5_fixed_rr_all", tp_mode="fixed_rr", reward_multiple=1.5),
        Params(name="bb_m5_tight_chop_all", adx_max=16.0, bb_width_mult_max=1.0, atr_expansion_max=1.1),
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
            "intrabar_order": "conservative: SL before TP if both touched",
            "live_side_effects": "none; script is research-only and reads CSV cache",
        },
        "grouped": {},
    }
    for params in variants:
        trades, metrics = backtest(df, params)
        summary.append({"strategy": params.name, **metrics})
        all_trades.extend(trades)
        diagnostics["grouped"][params.name] = {
            "side": grouped(trades, "side"),
            "session": grouped(trades, "session"),
        }
    write_outputs(summary, all_trades, diagnostics)
    print(json.dumps({"summary": summary, "out_dir": str(OUT_DIR)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
