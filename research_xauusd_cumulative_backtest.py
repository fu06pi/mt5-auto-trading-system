#!/usr/bin/env python3
"""Standard cumulative backtest for XAUUSD trend strategy.

Unlike the weekly-reset version, this lets equity compound naturally.
Reports: equity curve, max drawdown, Sharpe, Calmar, monthly returns, etc.

Uses the same data sources:
- MT5 broker M5 bars (what we have, ~74 weeks)
- Yahoo Finance GC=F daily (5-year proxy)
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path("/home/chain4655/Documents/Projects/MT5")
OUT_ROOT = Path("/home/chain4655/Documents/backtest_reports/xauusd_cumulative")
DATA_CACHE = Path("/home/chain4655/Documents/backtest_reports/xauusd_5y_weekly_reset/data/XAUUSD_M5_5y_mt5.csv")
PYMT5_PATH = Path("/home/chain4655/.openharness-venv/lib64/python3.14/site-packages")
ACTIVE_PLAN = ROOT / "auto_quant/active_plan.json"

CONTRACT_SIZE = 100.0
POINT = 0.01


def arg_value(cmd: Sequence[str], name: str, default: str) -> str:
    try:
        idx = list(cmd).index(name)
        return str(cmd[idx + 1])
    except (ValueError, IndexError):
        return default


@dataclasses.dataclass(frozen=True)
class Config:
    start_equity: float = 100000.0
    risk_pct: float = 0.0075
    daily_dd_limit: float = 0.030
    total_dd_limit: float = 0.045
    profit_target: float = 0.100
    max_lots: float = 1.2
    fast_sma: int = 20
    slow_sma: int = 60
    htf_fast_sma: int = 50
    htf_slow_sma: int = 200
    trend_threshold: float = 0.35
    atr_period: int = 14
    breakout_lookback: int = 20
    stop_atr: float = 2.5
    reward_multiple: float = 2.5
    primary_tp_reward_multiple: float = 1.2
    trail_trigger_atr: float = 1.5
    trail_lock_atr: float = 0.5
    break_even_atr: float = 1.0
    break_even_lock_atr: float = 0.15
    max_spread_points: float = 120.0
    max_trades_per_day: int = 999
    max_consecutive_losses: int = 9999
    loss_cooldown_losses: int = 3
    loss_cooldown_minutes: int = 60
    cooldown_bars_after_trade: int = 2
    startup_warmup_bars: int = 1
    max_hold_minutes: int = 180
    half_close_profit_usd: float = 2000.0
    half_close_fraction: float = 0.5
    half_close_cooldown_bars: int = 24
    warmup_risk_days: int = 7
    warmup_risk_multiplier: float = 0.5
    cost_per_lot_roundtrip: float = 40.0


def load_config() -> Config:
    plan = json.loads(ACTIVE_PLAN.read_text(encoding="utf-8"))
    cmd = list(plan.get("cmd") or [])
    return Config(
        start_equity=float(arg_value(cmd, "--start-equity", "100000")),
        risk_pct=float(arg_value(cmd, "--risk-pct", "0.0075")),
        daily_dd_limit=float(arg_value(cmd, "--daily-dd-limit", "0.030")),
        total_dd_limit=float(arg_value(cmd, "--total-dd-limit", "0.045")),
        profit_target=float(arg_value(cmd, "--profit-target", "0.100")),
        max_lots=float(arg_value(cmd, "--max-lots", "1.2")),
        fast_sma=int(arg_value(cmd, "--fast-sma", "20")),
        slow_sma=int(arg_value(cmd, "--slow-sma", "60")),
        htf_fast_sma=int(arg_value(cmd, "--htf-fast-sma", "50")),
        htf_slow_sma=int(arg_value(cmd, "--htf-slow-sma", "200")),
        trend_threshold=float(arg_value(cmd, "--trend-threshold", "0.35")),
        atr_period=int(arg_value(cmd, "--atr-period", "14")),
        breakout_lookback=int(arg_value(cmd, "--breakout-lookback", "20")),
        stop_atr=float(arg_value(cmd, "--stop-atr", "2.5")),
        reward_multiple=float(arg_value(cmd, "--reward-multiple", "2.5")),
        primary_tp_reward_multiple=float(arg_value(cmd, "--primary-tp-reward-multiple", "1.2")),
        trail_trigger_atr=float(arg_value(cmd, "--trail-trigger-atr", "1.5")),
        trail_lock_atr=float(arg_value(cmd, "--trail-lock-atr", "0.5")),
        break_even_atr=float(arg_value(cmd, "--break-even-atr", "1.0")),
        break_even_lock_atr=float(arg_value(cmd, "--break-even-lock-atr", "0.15")),
        max_spread_points=float(arg_value(cmd, "--max-spread-points", "120")),
        max_trades_per_day=int(arg_value(cmd, "--max-trades-per-day", "999")),
        max_consecutive_losses=int(arg_value(cmd, "--max-consecutive-losses", "9999")),
        loss_cooldown_losses=int(arg_value(cmd, "--loss-cooldown-losses", "3")),
        loss_cooldown_minutes=int(arg_value(cmd, "--loss-cooldown-minutes", "60")),
        cooldown_bars_after_trade=int(arg_value(cmd, "--cooldown-bars-after-trade", "2")),
        startup_warmup_bars=int(arg_value(cmd, "--startup-warmup-bars", "1")),
        max_hold_minutes=int(arg_value(cmd, "--max-hold-minutes", "180")),
        half_close_profit_usd=float(arg_value(cmd, "--auto-half-profit-usd", "2000")),
        half_close_fraction=float(arg_value(cmd, "--auto-half-fraction", "0.5")),
        half_close_cooldown_bars=int(arg_value(cmd, "--half-close-cooldown-bars", "24")),
        warmup_risk_days=int(arg_value(cmd, "--warmup-risk-days", "7")),
        warmup_risk_multiplier=float(arg_value(cmd, "--warmup-risk-multiplier", "0.5")),
    )


@dataclasses.dataclass
class Position:
    direction: str
    entry_time: pd.Timestamp
    entry: float
    sl: float
    tp: float
    volume: float
    half_done: bool = False


def round_volume(v: float) -> float:
    return max(0.01, math.floor(v / 0.01 + 1e-9) * 0.01)


def pnl_for(direction: str, entry: float, exit_price: float, volume: float, cost: float) -> float:
    gross = (exit_price - entry) * CONTRACT_SIZE * volume if direction == "BUY" else (entry - exit_price) * CONTRACT_SIZE * volume
    return gross - cost * volume


def floating_pnl(pos: Position, price: float) -> float:
    return (price - pos.entry) * CONTRACT_SIZE * pos.volume if pos.direction == "BUY" else (pos.entry - price) * CONTRACT_SIZE * pos.volume


def atr_sl_tp(direction: str, price: float, atr: float, cfg: Config) -> Tuple[float, float]:
    sl_dist = atr * cfg.stop_atr
    rr = cfg.primary_tp_reward_multiple if cfg.primary_tp_reward_multiple > 0 else cfg.reward_multiple
    tp_dist = sl_dist * rr
    if direction == "BUY":
        return round(price - sl_dist, 2), round(price + tp_dist, 2)
    return round(price + sl_dist, 2), round(price - tp_dist, 2)


def update_trail(pos: Position, close: float, atr: float, cfg: Config) -> None:
    if atr <= 0:
        return
    if pos.direction == "BUY":
        fav = close - pos.entry
        if fav >= cfg.break_even_atr * atr:
            pos.sl = max(pos.sl, round(pos.entry + cfg.break_even_lock_atr * atr, 2))
        if fav >= cfg.trail_trigger_atr * atr:
            pos.sl = max(pos.sl, round(close - cfg.trail_lock_atr * atr, 2))
    else:
        fav = pos.entry - close
        if fav >= cfg.break_even_atr * atr:
            pos.sl = min(pos.sl, round(pos.entry - cfg.break_even_lock_atr * atr, 2))
        if fav >= cfg.trail_trigger_atr * atr:
            pos.sl = min(pos.sl, round(close + cfg.trail_lock_atr * atr, 2))


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


def add_indicators_m5(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy().sort_values("time").reset_index(drop=True)
    out["time"] = pd.to_datetime(out["time"]).astype("datetime64[ns]")
    close = out["close"]
    high = out["high"]
    low = out["low"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(cfg.atr_period).mean().clip(lower=POINT * 5)
    out["fast_sma"] = close.rolling(cfg.fast_sma).mean()
    out["slow_sma"] = close.rolling(cfg.slow_sma).mean()
    out["momentum"] = ((close - close.shift(3)) / out["atr"]).clip(lower=-2.0, upper=2.0)
    out["recent_high"] = high.rolling(max(10, cfg.breakout_lookback)).max()
    out["recent_low"] = low.rolling(max(10, cfg.breakout_lookback)).min()
    h1 = (
        out.set_index("time")[["open", "high", "low", "close"]]
        .resample("1h", label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna().reset_index()
    )
    h1["htf_fast_sma"] = h1["close"].rolling(cfg.htf_fast_sma).mean()
    h1["htf_slow_sma"] = h1["close"].rolling(cfg.htf_slow_sma).mean()
    h1["htf_signal"] = "NEUTRAL"
    h1.loc[h1["htf_fast_sma"] > h1["htf_slow_sma"], "htf_signal"] = "BULL"
    h1.loc[h1["htf_fast_sma"] < h1["htf_slow_sma"], "htf_signal"] = "BEAR"
    h1["time"] = (h1["time"] + pd.Timedelta(microseconds=1)).astype("datetime64[ns]")
    out["time"] = out["time"].astype("datetime64[ns]")
    out = pd.merge_asof(out.sort_values("time"), h1[["time", "htf_fast_sma", "htf_slow_sma", "htf_signal"]].sort_values("time"), on="time", direction="backward")
    trend = pd.Series(0.0, index=out.index)
    trend[(out["close"] > out["fast_sma"]) & (out["fast_sma"] > out["slow_sma"])] = 0.55
    trend[(out["close"] < out["fast_sma"]) & (out["fast_sma"] < out["slow_sma"])] = -0.55
    trend[(trend == 0.0) & (out["close"] > out["slow_sma"])] = 0.20
    trend[(trend == 0.0) & (out["close"] < out["slow_sma"])] = -0.20
    htf_bias = pd.Series(0.0, index=out.index)
    htf_bias[out["htf_signal"] == "BULL"] = 0.35
    htf_bias[out["htf_signal"] == "BEAR"] = -0.35
    breakout = (((out["close"] - out["recent_high"]) / out["atr"]).clip(-1, 1) * 0.30) + (((out["recent_low"] - out["close"]) / out["atr"]).clip(-1, 1) * -0.30)
    out["score"] = (trend + htf_bias + breakout + out["momentum"].clip(-1.5, 1.5) * 0.25).clip(-1.5, 1.5)
    out["signal"] = "NONE"
    out.loc[(out["htf_signal"] == "BULL") & (out["score"] >= cfg.trend_threshold), "signal"] = "BUY"
    out.loc[(out["htf_signal"] == "BEAR") & (out["score"] <= -cfg.trend_threshold), "signal"] = "SELL"
    return out.dropna(subset=["atr", "fast_sma", "slow_sma", "htf_fast_sma", "htf_slow_sma"]).reset_index(drop=True)


def add_indicators_daily(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(cfg.atr_period).mean()
    out["fast_sma"] = close.rolling(cfg.fast_sma).mean()
    out["slow_sma"] = close.rolling(cfg.slow_sma).mean()
    out["momentum"] = ((close - close.shift(3)) / out["atr"]).clip(-2, 2)
    out["recent_high"] = high.rolling(max(10, cfg.breakout_lookback)).max()
    out["recent_low"] = low.rolling(max(10, cfg.breakout_lookback)).min()
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


def simulate_cumulative(df: pd.DataFrame, cfg: Config, bar_minutes: int = 5) -> Dict[str, object]:
    balance = cfg.start_equity
    max_equity = cfg.start_equity
    position: Optional[Position] = None
    trades: List[Dict[str, object]] = []
    equity_curve: List[Dict[str, object]] = []
    consecutive_losses = 0
    loss_cooldown_until: Optional[pd.Timestamp] = None
    last_trade_time: Optional[pd.Timestamp] = None
    last_half_close_time: Optional[pd.Timestamp] = None
    trades_today = 0
    current_day: Optional[dt.date] = None
    day_start_equity = cfg.start_equity
    warmup_seen = 0
    stopped_reason = ""

    def current_equity(price: float) -> float:
        return balance + (floating_pnl(position, price) if position else 0.0)

    def close_pos(t: pd.Timestamp, price: float, reason: str, volume_override: Optional[float] = None) -> float:
        nonlocal balance, position, consecutive_losses, loss_cooldown_until, last_half_close_time
        if position is None:
            return 0.0
        volume = position.volume if volume_override is None else min(position.volume, volume_override)
        gain = pnl_for(position.direction, position.entry, price, volume, cfg.cost_per_lot_roundtrip)
        balance += gain
        trades.append({
            "entry_time": position.entry_time.isoformat(),
            "exit_time": t.isoformat(),
            "direction": position.direction,
            "entry": round(position.entry, 2),
            "exit": round(price, 2),
            "volume": round(volume, 2),
            "pnl": round(gain, 2),
            "reason": reason,
        })
        if gain < 0:
            consecutive_losses += 1
            if consecutive_losses >= cfg.loss_cooldown_losses:
                loss_cooldown_until = t + pd.Timedelta(minutes=cfg.loss_cooldown_minutes)
        elif gain > 0:
            consecutive_losses = 0
            loss_cooldown_until = None
        if volume_override is None or volume >= position.volume - 1e-9:
            position = None
        else:
            position.volume = round_volume(position.volume - volume)
            position.half_done = True
            last_half_close_time = t
        return gain

    for idx, row in df.iterrows():
        t = pd.Timestamp(row["time"])
        close = float(row["close"])
        atr = float(row.get("atr", 0))
        if pd.isna(atr) or atr <= 0:
            continue

        if current_day != t.date():
            current_day = t.date()
            day_start_equity = current_equity(close)
            trades_today = 0

        # Manage open position
        if position is not None:
            update_trail(position, close, atr, cfg)
            reason, exit_price = exit_hit(position, row)
            if reason and exit_price is not None:
                close_pos(t, float(exit_price), reason)
            elif position is not None and (t - position.entry_time).total_seconds() / 60.0 >= cfg.max_hold_minutes:
                close_pos(t, close, "MAX_HOLD")
            elif position is not None and floating_pnl(position, close) >= cfg.half_close_profit_usd and not position.half_done:
                half_vol = round_volume(position.volume * cfg.half_close_fraction)
                if 0.01 <= half_vol < position.volume:
                    close_pos(t, close, "HALF_PROFIT", volume_override=half_vol)

        equity = current_equity(close)
        max_equity = max(max_equity, equity)

        # DD checks (but don't force-stop the whole run for cumulative)
        daily_dd = 1.0 - equity / max(day_start_equity, 1e-9)
        total_dd = 1.0 - equity / max(max_equity, 1e-9)

        # Record equity point
        equity_curve.append({"time": t.isoformat(), "equity": round(equity, 2), "drawdown_pct": round(total_dd * 100, 4)})

        # If total DD exceeds limit, close position and mark but CONTINUE running
        if total_dd >= cfg.total_dd_limit and position is not None:
            close_pos(t, close, "TOTAL_DD_HIT")
        if daily_dd >= cfg.daily_dd_limit and position is not None:
            close_pos(t, close, "DAILY_DD_HIT")

        # Entry logic
        if warmup_seen < cfg.startup_warmup_bars:
            warmup_seen += 1
            continue
        if position is not None:
            sig = str(row["signal"])
            if (position.direction == "BUY" and sig == "SELL") or (position.direction == "SELL" and sig == "BUY"):
                close_pos(t, close, "REVERSE")
            continue

        if position is None:
            if trades_today >= cfg.max_trades_per_day or consecutive_losses >= cfg.max_consecutive_losses:
                continue
            if loss_cooldown_until is not None and t < loss_cooldown_until:
                continue
            if last_half_close_time is not None and (t - last_half_close_time).total_seconds() < cfg.half_close_cooldown_bars * bar_minutes * 60:
                continue
            if last_trade_time is not None and (t - last_trade_time).total_seconds() < cfg.cooldown_bars_after_trade * bar_minutes * 60:
                continue
            sig = str(row["signal"])
            if sig in {"BUY", "SELL"}:
                sl, tp = atr_sl_tp(sig, close, atr, cfg)
                risk_per_lot = abs(close - sl) * CONTRACT_SIZE
                vol = round_volume(min(cfg.max_lots, (equity * cfg.risk_pct) / max(risk_per_lot, 1e-9)))
                if vol >= 0.01:
                    position = Position(direction=sig, entry_time=t, entry=close, sl=sl, tp=tp, volume=vol)
                    last_trade_time = t
                    trades_today += 1

    # Close any remaining position at end
    if position is not None and not df.empty:
        close_pos(pd.Timestamp(df.iloc[-1]["time"]), float(df.iloc[-1]["close"]), "END_OF_DATA")

    # Compute summary statistics
    final_equity = balance
    total_return = final_equity / cfg.start_equity - 1.0
    eq_series = pd.DataFrame(equity_curve)
    dd_series = eq_series["drawdown_pct"].astype(float)
    max_dd_pct = float(dd_series.max())
    gross_win = sum(float(t["pnl"]) for t in trades if float(t["pnl"]) > 0)
    gross_loss = -sum(float(t["pnl"]) for t in trades if float(t["pnl"]) < 0)
    wins = sum(1 for t in trades if float(t["pnl"]) > 0)
    losses_n = sum(1 for t in trades if float(t["pnl"]) < 0)

    # Monthly returns
    if trades:
        tr_df = pd.DataFrame(trades)
        tr_df["exit_month"] = pd.to_datetime(tr_df["exit_time"]).dt.to_period("M")
        monthly = tr_df.groupby("exit_month")["pnl"].sum().astype(float)
        monthly_ret = (monthly / cfg.start_equity * 100).round(4)
    else:
        monthly_ret = pd.Series(dtype=float)

    # Daily returns from equity curve
    eq_series["date"] = pd.to_datetime(eq_series["time"]).dt.date
    daily_eq = eq_series.groupby("date")["equity"].last()
    daily_ret = daily_eq.pct_change().dropna() * 100

    sharpe_daily = 0.0
    if len(daily_ret) > 1 and daily_ret.std() > 0:
        sharpe_daily = round((daily_ret.mean() / daily_ret.std()) * math.sqrt(252), 3)

    calmar = 0.0
    if max_dd_pct > 0:
        calmar = round(total_return * 100 / max_dd_pct, 3)

    return {
        "data_start": pd.Timestamp(df["time"].iloc[0]).isoformat(),
        "data_end": pd.Timestamp(df["time"].iloc[-1]).isoformat(),
        "bars": len(df),
        "start_equity": cfg.start_equity,
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(total_return * 100, 4),
        "max_drawdown_pct": round(max_dd_pct, 4),
        "calmar_ratio": calmar,
        "sharpe_annualized": sharpe_daily,
        "trades": len(trades),
        "wins": wins,
        "losses": losses_n,
        "trade_win_rate_pct": round(wins / len(trades) * 100, 2) if trades else 0.0,
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 0 else 999.0,
        "avg_trade_pnl": round(statistics.fmean([float(t["pnl"]) for t in trades]), 2) if trades else 0.0,
        "avg_win": round(gross_win / wins, 2) if wins else 0.0,
        "avg_loss": round(gross_loss / losses_n, 2) if losses_n else 0.0,
        "win_loss_ratio": round((gross_win / wins) / (gross_loss / losses_n), 3) if wins and losses_n else 999.0,
        "sl_hits": sum(1 for t in trades if t["reason"] == "SL"),
        "tp_hits": sum(1 for t in trades if t["reason"] == "TP"),
        "trail_close": sum(1 for t in trades if t["reason"] in {"REVERSE", "MAX_HOLD", "WEEK_END", "END_OF_DATA"}),
        "half_profit": sum(1 for t in trades if t["reason"] == "HALF_PROFIT"),
        "dd_hit_trades": sum(1 for t in trades if "DD" in str(t["reason"])),
        "monthly_returns_pct": {str(k): v for k, v in monthly_ret.items()},
        "equity_curve_file": "equity_curve.csv",
        "trades_file": "trades.csv",
    }, equity_curve, trades


def flatten_yf(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(c[0]).lower().replace(" ", "_") for c in out.columns]
    else:
        out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    return out.rename(columns={"date": "time", "datetime": "time"})


def main() -> int:
    cfg = load_config()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # --- MT5 M5 data ---
    if DATA_CACHE.exists():
        raw_m5 = pd.read_csv(DATA_CACHE, parse_dates=["time"])
        df_m5 = add_indicators_m5(raw_m5, cfg)
        summary_m5, eq_m5, tr_m5 = simulate_cumulative(df_m5, cfg, bar_minutes=5)
        summary_m5["source"] = "MT5_M5_broker"
        results["mt5_m5"] = summary_m5
        pd.DataFrame(eq_m5).to_csv(out_dir / "mt5_m5_equity_curve.csv", index=False)
        pd.DataFrame(tr_m5).to_csv(out_dir / "mt5_m5_trades.csv", index=False)
        (out_dir / "mt5_m5_summary.json").write_text(json.dumps(summary_m5, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"MT5 M5 done: {summary_m5['total_return_pct']}% return, {summary_m5['max_drawdown_pct']}% max DD")
    else:
        print("No MT5 M5 cache found, skipping")

    # --- Yahoo daily proxy 5y ---
    import yfinance as yf
    raw_d = yf.download("GC=F", period="5y", interval="1d", auto_adjust=False, progress=False)
    if not raw_d.empty:
        daily = flatten_yf(raw_d)
        daily = daily[["time", "open", "high", "low", "close", "volume"]].dropna()
        daily["time"] = pd.to_datetime(daily["time"]).dt.tz_localize(None)
        daily = daily.sort_values("time").reset_index(drop=True)
        df_d = add_indicators_daily(daily, cfg)
        summary_d, eq_d, tr_d = simulate_cumulative(df_d, cfg, bar_minutes=1440)
        summary_d["source"] = "Yahoo_GC=F_daily_5y_proxy"
        results["daily_proxy"] = summary_d
        pd.DataFrame(eq_d).to_csv(out_dir / "daily_proxy_equity_curve.csv", index=False)
        pd.DataFrame(tr_d).to_csv(out_dir / "daily_proxy_trades.csv", index=False)
        (out_dir / "daily_proxy_summary.json").write_text(json.dumps(summary_d, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Daily proxy done: {summary_d['total_return_pct']}% return, {summary_d['max_drawdown_pct']}% max DD")

    (out_dir / "all_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
