#!/usr/bin/env python3
"""Research-only TradingView built-in strategy style backtest on MT5 cached bars.

This ports the visible TradingView strategy-list ideas into simple non-repainting
Python signal functions. It never places orders and never edits active_plan.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path("/home/chain4655/Documents/Projects/MT5")
DEFAULT_DATA = Path("/home/chain4655/Documents/backtest_reports/xauusd_5y_weekly_reset/data/XAUUSD_M5_5y_mt5.csv")
ACTIVE_PLAN = ROOT / "auto_quant/active_plan.json"
OUT_DIR = ROOT / "backtest_reports_tv_builtin"
INITIAL_EQUITY = 100000.0
CONTRACT_SIZE = 100.0
POINT = 0.01
SPREAD_POINTS = 49.0
COMMISSION_PER_LOT = 7.0


@dataclass(frozen=True)
class Params:
    name: str
    family: str
    trade_type: str = "BOTH"  # LONG, SHORT, BOTH
    risk_pct: float = 0.001
    max_lots: float = 0.5
    sl_atr: float = 1.5
    reward_r: float = 2.0
    max_hold_bars: int = 72
    length: int = 20
    length2: int = 50
    mult: float = 2.0
    min_adx: float = 0.0
    max_adx: float = 999.0
    htf_gate: str = "NONE"  # NONE, RAW
    range_only: bool = False
    session_start_hour: int = 0
    session_end_hour: int = 24
    one_signal_per_session: bool = False


@dataclass(frozen=True)
class Trade:
    strategy: str
    family: str
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
    adx: float
    signal_time: str


def sha256_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bars(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"Time": "time", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "tick_volume"})
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
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def fetch_mt5_bars(symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    from pymt5linux import MetaTrader5

    tf_map = {
        "M1": MetaTrader5.TIMEFRAME_M1,
        "M5": MetaTrader5.TIMEFRAME_M5,
        "M15": MetaTrader5.TIMEFRAME_M15,
        "M30": MetaTrader5.TIMEFRAME_M30,
        "H1": MetaTrader5.TIMEFRAME_H1,
    }
    mt5 = MetaTrader5(host="127.0.0.1", port=18812)
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"copy_rates_from_pos failed: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    if "time" not in df.columns:
        df = pd.DataFrame([r._asdict() if hasattr(r, "_asdict") else dict(r) for r in rates])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df[["time", "open", "high", "low", "close", "tick_volume"]].sort_values("time").reset_index(drop=True)


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    high = out["high"]
    low = out["low"]
    close = out["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(14).mean()
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr_sum = tr.rolling(14).sum().mask(lambda s: s == 0, math.nan)
    plus_di = 100.0 * plus_dm.rolling(14).sum() / tr_sum
    minus_di = 100.0 * minus_dm.rolling(14).sum() / tr_sum
    denom = (plus_di + minus_di).mask(lambda s: s == 0, math.nan)
    out["adx"] = (100.0 * (plus_di - minus_di).abs() / denom).rolling(14).mean().fillna(50.0)

    out["ema20"] = ema(close, 20)
    out["ema50"] = ema(close, 50)
    out["ema200"] = ema(close, 200)
    out["htf_raw"] = "NONE"
    out.loc[out["ema50"] > out["ema200"], "htf_raw"] = "BULL"
    out.loc[out["ema50"] < out["ema200"], "htf_raw"] = "BEAR"

    basis = close.rolling(20).mean()
    std = close.rolling(20).std()
    out["bb_mid"] = basis
    out["bb_upper"] = basis + 2.0 * std
    out["bb_lower"] = basis - 2.0 * std
    out["kc_mid"] = ema(close, 20)
    out["kc_upper"] = out["kc_mid"] + 2.0 * out["atr"]
    out["kc_lower"] = out["kc_mid"] - 2.0 * out["atr"]
    exp1 = ema(close, 12)
    exp2 = ema(close, 26)
    out["macd"] = exp1 - exp2
    out["macd_signal"] = ema(out["macd"], 9)
    out["mom10"] = close - close.shift(10)
    out["sma_fast"] = close.rolling(20).mean()
    out["sma_slow"] = close.rolling(50).mean()
    return out


def add_signals(df: pd.DataFrame, params: Params) -> pd.DataFrame:
    out = add_indicators(df)
    out["raw_signal"] = ""
    c = out["close"]
    o = out["open"]
    h = out["high"]
    l = out["low"]
    fam = params.family

    if fam == "bar_updn":
        out.loc[(c > o) & (c > c.shift(1)), "raw_signal"] = "BUY"
        out.loc[(c < o) & (c < c.shift(1)), "raw_signal"] = "SELL"
    elif fam == "bb_mean_reversion":
        out.loc[(c.shift(1) < out["bb_lower"].shift(1)) & (c > out["bb_lower"]), "raw_signal"] = "BUY"
        out.loc[(c.shift(1) > out["bb_upper"].shift(1)) & (c < out["bb_upper"]), "raw_signal"] = "SELL"
    elif fam == "bb_directional":
        out.loc[(c > out["bb_upper"]) & (c.shift(1) <= out["bb_upper"].shift(1)), "raw_signal"] = "BUY"
        out.loc[(c < out["bb_lower"]) & (c.shift(1) >= out["bb_lower"].shift(1)), "raw_signal"] = "SELL"
    elif fam == "channel_breakout":
        upper = h.shift(1).rolling(params.length).max()
        lower = l.shift(1).rolling(params.length).min()
        out.loc[c > upper, "raw_signal"] = "BUY"
        out.loc[c < lower, "raw_signal"] = "SELL"
    elif fam == "consecutive_updown":
        up = (c > c.shift(1)).rolling(params.length).sum()
        down = (c < c.shift(1)).rolling(params.length).sum()
        out.loc[up >= params.length, "raw_signal"] = "BUY"
        out.loc[down >= params.length, "raw_signal"] = "SELL"
    elif fam == "inside_bar":
        mother_h = h.shift(2)
        mother_l = l.shift(2)
        inside_prev = (h.shift(1) < mother_h) & (l.shift(1) > mother_l)
        out.loc[inside_prev & (c > mother_h), "raw_signal"] = "BUY"
        out.loc[inside_prev & (c < mother_l), "raw_signal"] = "SELL"
    elif fam == "keltner_directional":
        out.loc[(c > out["kc_upper"]) & (c.shift(1) <= out["kc_upper"].shift(1)), "raw_signal"] = "BUY"
        out.loc[(c < out["kc_lower"]) & (c.shift(1) >= out["kc_lower"].shift(1)), "raw_signal"] = "SELL"
    elif fam == "macd":
        out.loc[(out["macd"] > out["macd_signal"]) & (out["macd"].shift(1) <= out["macd_signal"].shift(1)), "raw_signal"] = "BUY"
        out.loc[(out["macd"] < out["macd_signal"]) & (out["macd"].shift(1) >= out["macd_signal"].shift(1)), "raw_signal"] = "SELL"
    elif fam == "momentum":
        out.loc[(out["mom10"] > 0) & (out["mom10"].shift(1) <= 0), "raw_signal"] = "BUY"
        out.loc[(out["mom10"] < 0) & (out["mom10"].shift(1) >= 0), "raw_signal"] = "SELL"
    elif fam == "ma_cross":
        out.loc[(out["sma_fast"] > out["sma_slow"]) & (out["sma_fast"].shift(1) <= out["sma_slow"].shift(1)), "raw_signal"] = "BUY"
        out.loc[(out["sma_fast"] < out["sma_slow"]) & (out["sma_fast"].shift(1) >= out["sma_slow"].shift(1)), "raw_signal"] = "SELL"
    elif fam == "outside_bar":
        outside = (h > h.shift(1)) & (l < l.shift(1))
        out.loc[outside & (c > o), "raw_signal"] = "BUY"
        out.loc[outside & (c < o), "raw_signal"] = "SELL"
    else:
        raise ValueError(f"unknown family: {fam}")

    if params.htf_gate == "RAW":
        out.loc[(out["raw_signal"] == "BUY") & (out["htf_raw"] != "BULL"), "raw_signal"] = ""
        out.loc[(out["raw_signal"] == "SELL") & (out["htf_raw"] != "BEAR"), "raw_signal"] = ""
    if params.trade_type == "LONG":
        out.loc[out["raw_signal"] == "SELL", "raw_signal"] = ""
    elif params.trade_type == "SHORT":
        out.loc[out["raw_signal"] == "BUY", "raw_signal"] = ""
    if params.range_only:
        out.loc[out["adx"] > params.max_adx, "raw_signal"] = ""
    out.loc[out["adx"] < params.min_adx, "raw_signal"] = ""
    if params.session_start_hour != 0 or params.session_end_hour != 24:
        hour = out["time"].dt.hour
        if params.session_start_hour < params.session_end_hour:
            ok = (hour >= params.session_start_hour) & (hour < params.session_end_hour)
        else:
            ok = (hour >= params.session_start_hour) | (hour < params.session_end_hour)
        out.loc[~ok, "raw_signal"] = ""
    return out


def calc_lots(equity: float, risk_pct: float, risk_distance: float, max_lots: float) -> float:
    if risk_distance <= 0:
        return 0.0
    lots = equity * risk_pct / (risk_distance * CONTRACT_SIZE)
    lots = min(max_lots, math.floor(lots * 100.0) / 100.0)
    return max(0.01, lots) if lots >= 0.01 else 0.0


def close_trade(pos: Dict[str, Any], row: pd.Series, exit_price: float, reason: str, i: int) -> Trade:
    direction = 1.0 if pos["side"] == "BUY" else -1.0
    gross = (exit_price - pos["entry"]) * direction * CONTRACT_SIZE * pos["volume"]
    spread_cost = SPREAD_POINTS * POINT * CONTRACT_SIZE * pos["volume"]
    commission = COMMISSION_PER_LOT * pos["volume"]
    net = gross - spread_cost - commission
    risk_cash = abs(pos["entry"] - pos["sl"]) * CONTRACT_SIZE * pos["volume"]
    return Trade(
        strategy=str(pos["strategy"]), family=str(pos["family"]), entry_time=str(pos["entry_time"]),
        exit_time=str(row["time"]), side=str(pos["side"]), entry=float(pos["entry"]), exit=float(exit_price),
        sl=float(pos["sl"]), tp=float(pos["tp"]), volume=float(pos["volume"]), gross_pnl=float(gross),
        net_pnl=float(net), commission=float(commission), spread_cost=float(spread_cost),
        r_multiple=float(net / risk_cash) if risk_cash > 0 else 0.0, reason=reason,
        bars_held=i - int(pos["entry_idx"]), atr=float(pos["atr"]), adx=float(pos["adx"]),
        signal_time=str(pos["signal_time"]),
    )


def backtest(df: pd.DataFrame, params: Params) -> Tuple[List[Trade], Dict[str, Any]]:
    data = add_signals(df, params).dropna(subset=["atr", "adx"]).reset_index(drop=True)
    trades: List[Trade] = []
    equity = INITIAL_EQUITY
    peak = equity
    max_dd = 0.0
    pos: Optional[Dict[str, Any]] = None
    used_session_keys: set[tuple[str, str, str]] = set()

    for i in range(2, len(data)):
        row = data.iloc[i]
        prev = data.iloc[i - 1]
        signal = str(prev["raw_signal"] or "")
        if pos is not None:
            exit_price = None
            reason = ""
            if pos["side"] == "BUY":
                if row["low"] <= pos["sl"]:
                    exit_price, reason = pos["sl"], "SL"
                elif row["high"] >= pos["tp"]:
                    exit_price, reason = pos["tp"], "TP"
                elif signal == "SELL":
                    exit_price, reason = float(row["open"]), "REVERSE"
            else:
                if row["high"] >= pos["sl"]:
                    exit_price, reason = pos["sl"], "SL"
                elif row["low"] <= pos["tp"]:
                    exit_price, reason = pos["tp"], "TP"
                elif signal == "BUY":
                    exit_price, reason = float(row["open"]), "REVERSE"
            if exit_price is None and i - int(pos["entry_idx"]) >= params.max_hold_bars:
                exit_price, reason = float(row["close"]), "MAX_HOLD"
            if exit_price is not None:
                trade = close_trade(pos, row, float(exit_price), reason, i)
                trades.append(trade)
                equity += trade.net_pnl
                peak = max(peak, equity)
                max_dd = max(max_dd, (peak - equity) / peak if peak > 0 else 0.0)
                pos = None

        if pos is None and signal in {"BUY", "SELL"}:
            if params.one_signal_per_session:
                session_key = (str(row["time"].date()), signal, params.name)
                if session_key in used_session_keys:
                    continue
                used_session_keys.add(session_key)
            if row["atr"] <= 0:
                continue
            entry = float(row["open"])
            risk_distance = float(row["atr"]) * params.sl_atr
            if signal == "BUY":
                sl = entry - risk_distance
                tp = entry + risk_distance * params.reward_r
            else:
                sl = entry + risk_distance
                tp = entry - risk_distance * params.reward_r
            volume = calc_lots(equity, params.risk_pct, risk_distance, params.max_lots)
            if volume <= 0:
                continue
            pos = {
                "strategy": params.name, "family": params.family, "entry_idx": i, "entry_time": row["time"],
                "signal_time": prev["time"], "side": signal, "entry": entry, "sl": sl, "tp": tp,
                "volume": volume, "atr": float(row["atr"]), "adx": float(row["adx"]),
            }

    if pos is not None and len(data) > 0:
        row = data.iloc[-1]
        trade = close_trade(pos, row, float(row["close"]), "EOD", len(data) - 1)
        trades.append(trade)
        equity += trade.net_pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak if peak > 0 else 0.0)

    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]
    gross_profit = sum(t.net_pnl for t in wins)
    gross_loss = -sum(t.net_pnl for t in losses)
    summary: Dict[str, Any] = {
        "strategy": params.name, "family": params.family, "params": asdict(params),
        "start": str(data["time"].iloc[0]) if len(data) else None,
        "end": str(data["time"].iloc[-1]) if len(data) else None,
        "bars": int(len(data)), "trades": len(trades),
        "win_rate": round(len(wins) / len(trades), 4) if trades else 0.0,
        "net_pnl": round(sum(t.net_pnl for t in trades), 2),
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss > 0 else None,
        "max_closed_dd_pct": round(max_dd * 100.0, 2),
        "avg_r": round(sum(t.r_multiple for t in trades) / len(trades), 3) if trades else 0.0,
        "final_equity": round(equity, 2), "exit_reasons": {},
    }
    for t in trades:
        summary["exit_reasons"][t.reason] = summary["exit_reasons"].get(t.reason, 0) + 1
    return trades, summary


def default_variants() -> List[Params]:
    return [
        Params("TV_ChannelBreakout_raw_20", "channel_breakout", length=20, htf_gate="RAW", min_adx=24, reward_r=2.0),
        Params("TV_ChannelBreakout_buy_raw_20", "channel_breakout", trade_type="LONG", length=20, htf_gate="RAW", min_adx=24, reward_r=2.0),
        Params("TV_ChannelBreakout_buy_h1_6", "channel_breakout", trade_type="LONG", length=20, htf_gate="RAW", min_adx=24, reward_r=2.0, session_start_hour=1, session_end_hour=6),
        Params("TV_ChannelBreakout_buy_h9_10", "channel_breakout", trade_type="LONG", length=20, htf_gate="RAW", min_adx=24, reward_r=2.0, session_start_hour=9, session_end_hour=10),
        Params("TV_ChannelBreakout_buy_h15_16", "channel_breakout", trade_type="LONG", length=20, htf_gate="RAW", min_adx=24, reward_r=2.0, session_start_hour=15, session_end_hour=16),
        Params("TV_ChannelBreakout_early_one_shot", "channel_breakout", length=20, htf_gate="RAW", min_adx=24, session_start_hour=13, session_end_hour=20, one_signal_per_session=True),
        Params("TV_BB_Directional_raw", "bb_directional", htf_gate="RAW", min_adx=24, reward_r=1.8),
        Params("TV_Keltner_Directional_raw", "keltner_directional", htf_gate="RAW", min_adx=24, reward_r=1.8),
        Params("TV_InsideBar_raw", "inside_bar", htf_gate="RAW", min_adx=18, reward_r=1.8),
        Params("TV_OutsideBar_raw", "outside_bar", htf_gate="RAW", min_adx=18, reward_r=1.5),
        Params("TV_BB_MeanReversion_range", "bb_mean_reversion", range_only=True, max_adx=18, reward_r=1.2, max_hold_bars=36),
        Params("TV_Consecutive3_raw", "consecutive_updown", length=3, htf_gate="RAW", min_adx=20, reward_r=1.5),
        Params("TV_BarUpDn_raw", "bar_updn", htf_gate="RAW", min_adx=24, reward_r=1.2, max_hold_bars=36),
        Params("TV_MACD_raw", "macd", htf_gate="RAW", min_adx=20, reward_r=1.8),
        Params("TV_Momentum_raw", "momentum", htf_gate="RAW", min_adx=20, reward_r=1.5),
        Params("TV_MA_Cross_raw", "ma_cross", htf_gate="RAW", min_adx=20, reward_r=2.0, max_hold_bars=120),
    ]


def write_outputs(trades: List[Trade], summaries: List[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(t) for t in trades]).to_csv(OUT_DIR / "tv_builtin_trades.csv", index=False)
    pd.DataFrame(summaries).sort_values(["profit_factor", "net_pnl"], ascending=[False, False]).to_csv(OUT_DIR / "tv_builtin_summary.csv", index=False)
    (OUT_DIR / "tv_builtin_summary.json").write_text(json.dumps({"metadata": metadata, "summaries": summaries}, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--fetch-mt5", action="store_true")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M5")
    parser.add_argument("--bars", type=int, default=50000)
    args = parser.parse_args()

    plan_before = sha256_file(ACTIVE_PLAN)
    if args.fetch_mt5:
        df = fetch_mt5_bars(args.symbol, args.timeframe, args.bars)
    else:
        df = load_bars(args.data)

    summaries: List[Dict[str, Any]] = []
    all_trades: List[Trade] = []
    for params in default_variants():
        trades, summary = backtest(df, params)
        summaries.append(summary)
        all_trades.extend(trades)
    plan_after = sha256_file(ACTIVE_PLAN)
    metadata = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "data_start": str(df["time"].iloc[0]),
        "data_end": str(df["time"].iloc[-1]),
        "input_bars": int(len(df)),
        "active_plan_sha256_before": plan_before,
        "active_plan_sha256_after": plan_after,
        "active_plan_unchanged": plan_before == plan_after,
        "output_dir": str(OUT_DIR),
        "note": "Research-only non-repainting approximations of visible TradingView built-in strategy ideas.",
    }
    write_outputs(all_trades, summaries, metadata)
    ranked = sorted(summaries, key=lambda s: ((s["profit_factor"] or 0), s["net_pnl"]), reverse=True)
    print(json.dumps({"metadata": metadata, "top": ranked[:8], "all_count": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
