#!/usr/bin/env python3
"""Five-year XAUUSD weekly-reset backtest for the active trend strategy.

This research script intentionally does not import the live strategy class because
that class is tightly coupled to MT5 live state.  Instead it ports the active plan's
signal/risk rules into a deterministic bar-by-bar simulator:

- M5 XAUUSD bars from broker MT5/pymt5linux, cached as CSV.
- H1 HTF filter built from M5 resampling.
- Each ISO week starts with a fresh account balance (default 100k).
- Open positions are closed at week end and weekly return is recorded as a sample.
- Daily DD / total DD / profit target are evaluated inside each weekly episode.

Outputs are written to /home/chain4655/Documents/backtest_reports/xauusd_5y_weekly_reset/<timestamp>/.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path("/home/chain4655/Documents/Projects/MT5")
OUT_ROOT = Path("/home/chain4655/Documents/backtest_reports/xauusd_5y_weekly_reset")
DATA_ROOT = OUT_ROOT / "data"
ACTIVE_PLAN = ROOT / "auto_quant/active_plan.json"
PYMT5_PATH = Path("/home/chain4655/.openharness-venv/lib64/python3.14/site-packages")

SYMBOL = "XAUUSD"
POINT = 0.01
CONTRACT_SIZE = 100.0


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


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
    cost_per_lot_roundtrip: float = 40.0  # ~35-point spread + ~$5 commission per lot.


@dataclasses.dataclass
class Position:
    direction: str
    entry_time: pd.Timestamp
    entry: float
    sl: float
    tp: float
    volume: float
    half_done: bool = False


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


def import_pymt5():
    sys.path.insert(0, str(PYMT5_PATH))
    from pymt5linux import MetaTrader5  # type: ignore

    return MetaTrader5


def rates_to_df(rates: object) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for row in list(rates):
        def get(name: str, idx: int) -> float:
            try:
                return float(row[name])  # structured numpy row
            except Exception:
                try:
                    return float(getattr(row, name))
                except Exception:
                    return float(row[idx])

        ts = int(get("time", 0))
        rows.append(
            {
                "time": pd.to_datetime(ts, unit="s", utc=True).tz_convert(None),
                "open": get("open", 1),
                "high": get("high", 2),
                "low": get("low", 3),
                "close": get("close", 4),
                "tick_volume": get("tick_volume", 5),
            }
        )
    df = pd.DataFrame(rows).drop_duplicates("time").sort_values("time")
    return df.reset_index(drop=True)


def fetch_m5_from_mt5(symbol: str, start: dt.datetime, end: dt.datetime, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["time"])
        if not cached.empty and cached["time"].min() <= pd.Timestamp(start) and cached["time"].max() >= pd.Timestamp(end) - pd.Timedelta(days=2):
            return cached

    MetaTrader5 = import_pymt5()

    def fetch_chunk(chunk_start: dt.datetime, chunk_end: dt.datetime) -> Optional[pd.DataFrame]:
        for attempt in range(1, 4):
            mt5 = MetaTrader5(host="127.0.0.1", port=18812)
            if not mt5.initialize():
                err = mt5.last_error()
                try:
                    mt5.shutdown()
                except Exception:
                    pass
                if attempt == 3:
                    print(f"WARN: init failed chunk {chunk_start.isoformat()} -> {chunk_end.isoformat()} err={err}", file=sys.stderr)
                    return None
                time.sleep(1)
                continue
            try:
                mt5.symbol_select(symbol, True)
                # Verify IPC with a tiny probe before the larger history call.
                if mt5.symbol_info(symbol) is None:
                    raise RuntimeError(f"symbol_info failed: {mt5.last_error()}")
                rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, chunk_start, chunk_end)
                if rates is not None and len(rates) > 0:
                    return rates_to_df(rates)
                err = mt5.last_error()
                if attempt == 3:
                    print(f"WARN: empty chunk {chunk_start.isoformat()} -> {chunk_end.isoformat()} err={err}", file=sys.stderr)
                    return None
            except Exception as exc:
                if attempt == 3:
                    print(f"WARN: exception chunk {chunk_start.isoformat()} -> {chunk_end.isoformat()} exc={exc}", file=sys.stderr)
                    return None
            finally:
                try:
                    mt5.shutdown()
                except Exception:
                    pass
            time.sleep(1)
        return None

    frames: List[pd.DataFrame] = []
    cursor = start
    chunk_days = 90
    while cursor < end:
        chunk_end = min(cursor + dt.timedelta(days=chunk_days), end)
        frame = fetch_chunk(cursor, chunk_end)
        if frame is not None and not frame.empty:
            frames.append(frame)
        if chunk_end >= end:
            break
        cursor = chunk_end  # no overlap; duplicates are dropped later.
        time.sleep(0.1)

    if not frames:
        raise RuntimeError("No MT5 M5 rates returned in any chunk")
    df = pd.concat(frames, ignore_index=True).drop_duplicates("time").sort_values("time").reset_index(drop=True)

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def add_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
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
        .dropna()
        .reset_index()
    )
    h1["htf_fast_sma"] = h1["close"].rolling(cfg.htf_fast_sma).mean()
    h1["htf_slow_sma"] = h1["close"].rolling(cfg.htf_slow_sma).mean()
    h1["htf_signal"] = "NEUTRAL"
    h1.loc[h1["htf_fast_sma"] > h1["htf_slow_sma"], "htf_signal"] = "BULL"
    h1.loc[h1["htf_fast_sma"] < h1["htf_slow_sma"], "htf_signal"] = "BEAR"
    # Use only the last fully closed H1 bar at each M5 close.
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
    momentum_component = out["momentum"].clip(-1.5, 1.5) * 0.25
    out["score"] = (trend + htf_bias + breakout + momentum_component).clip(-1.5, 1.5)
    out["signal"] = "NONE"
    out.loc[(out["htf_signal"] == "BULL") & (out["score"] >= cfg.trend_threshold), "signal"] = "BUY"
    out.loc[(out["htf_signal"] == "BEAR") & (out["score"] <= -cfg.trend_threshold), "signal"] = "SELL"
    return out


def pnl_for(direction: str, entry: float, exit_price: float, volume: float, cost_per_lot: float) -> float:
    if direction == "BUY":
        gross = (exit_price - entry) * CONTRACT_SIZE * volume
    else:
        gross = (entry - exit_price) * CONTRACT_SIZE * volume
    return gross - (cost_per_lot * volume)


def floating_pnl(pos: Position, price: float) -> float:
    if pos.direction == "BUY":
        return (price - pos.entry) * CONTRACT_SIZE * pos.volume
    return (pos.entry - price) * CONTRACT_SIZE * pos.volume


def round_volume(volume: float) -> float:
    return max(0.01, math.floor(volume / 0.01 + 1e-9) * 0.01)


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
        favorable = close - pos.entry
        if favorable >= cfg.break_even_atr * atr:
            new_sl = round(pos.entry + cfg.break_even_lock_atr * atr, 2)
            pos.sl = max(pos.sl, new_sl)
        if favorable >= cfg.trail_trigger_atr * atr:
            new_sl = round(close - cfg.trail_lock_atr * atr, 2)
            pos.sl = max(pos.sl, new_sl)
    else:
        favorable = pos.entry - close
        if favorable >= cfg.break_even_atr * atr:
            new_sl = round(pos.entry - cfg.break_even_lock_atr * atr, 2)
            pos.sl = min(pos.sl, new_sl)
        if favorable >= cfg.trail_trigger_atr * atr:
            new_sl = round(close + cfg.trail_lock_atr * atr, 2)
            pos.sl = min(pos.sl, new_sl)


def exit_hit(pos: Position, row: pd.Series) -> Tuple[Optional[str], Optional[float]]:
    high = float(row["high"])
    low = float(row["low"])
    if pos.direction == "BUY":
        # Conservative: if both hit in one M5 bar, assume SL first.
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


def effective_risk_pct(week_start: pd.Timestamp, t: pd.Timestamp, cfg: Config, reset_warmup: bool) -> float:
    if not reset_warmup or cfg.warmup_risk_days <= 0:
        return cfg.risk_pct
    if t < week_start + pd.Timedelta(days=cfg.warmup_risk_days):
        return cfg.risk_pct * cfg.warmup_risk_multiplier
    return cfg.risk_pct


def simulate_week(week_df: pd.DataFrame, cfg: Config, reset_warmup: bool) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    start_equity = cfg.start_equity
    balance = start_equity
    day_start_equity = start_equity
    max_equity = start_equity
    trades_today = 0
    current_day: Optional[dt.date] = None
    consecutive_losses = 0
    loss_cooldown_until: Optional[pd.Timestamp] = None
    last_trade_time: Optional[pd.Timestamp] = None
    last_half_close_time: Optional[pd.Timestamp] = None
    position: Optional[Position] = None
    trades: List[Dict[str, object]] = []
    stopped_reason = ""
    warmup_seen = 0
    week_start = pd.Timestamp(week_df["time"].iloc[0]).normalize()

    def current_equity(price: float) -> float:
        return balance + (floating_pnl(position, price) if position else 0.0)

    def close_position(t: pd.Timestamp, price: float, reason: str, row: pd.Series, volume_override: Optional[float] = None) -> float:
        nonlocal balance, position, consecutive_losses, loss_cooldown_until, last_half_close_time
        if position is None:
            return 0.0
        volume = position.volume if volume_override is None else min(position.volume, volume_override)
        pnl = pnl_for(position.direction, position.entry, price, volume, cfg.cost_per_lot_roundtrip)
        balance += pnl
        trades.append(
            {
                "week": str(row["week"]),
                "entry_time": position.entry_time.isoformat(),
                "exit_time": t.isoformat(),
                "direction": position.direction,
                "entry": round(position.entry, 2),
                "exit": round(price, 2),
                "volume": round(volume, 2),
                "pnl": round(pnl, 2),
                "reason": reason,
            }
        )
        if pnl < 0:
            consecutive_losses += 1
            if consecutive_losses >= cfg.loss_cooldown_losses:
                loss_cooldown_until = t + pd.Timedelta(minutes=cfg.loss_cooldown_minutes)
        elif pnl > 0:
            consecutive_losses = 0
            loss_cooldown_until = None
        if volume_override is None or volume >= position.volume - 1e-9:
            position = None
        else:
            position.volume = round_volume(position.volume - volume)
            position.half_done = True
            last_half_close_time = t
        return pnl

    for _, row in week_df.iterrows():
        if pd.isna(row["atr"]) or pd.isna(row["htf_slow_sma"]):
            continue
        t = pd.Timestamp(row["time"])
        close = float(row["close"])
        atr = float(row["atr"])
        if current_day != t.date():
            current_day = t.date()
            day_start_equity = current_equity(close)
            trades_today = 0

        # Manage open position first.
        if position is not None:
            update_trail(position, close, atr, cfg)
            reason, exit_price = exit_hit(position, row)
            if reason and exit_price is not None:
                close_position(t, float(exit_price), reason, row)
            elif position is not None and (t - position.entry_time).total_seconds() / 60.0 >= cfg.max_hold_minutes:
                close_position(t, close, "MAX_HOLD", row)
            elif position is not None and floating_pnl(position, close) >= cfg.half_close_profit_usd and not position.half_done:
                half_volume = round_volume(position.volume * cfg.half_close_fraction)
                if 0.01 <= half_volume < position.volume:
                    close_position(t, close, "HALF_PROFIT", row, volume_override=half_volume)

        equity = current_equity(close)
        max_equity = max(max_equity, equity)
        daily_dd = 1.0 - equity / max(day_start_equity, 1e-9)
        total_dd = 1.0 - equity / max(max_equity, 1e-9)
        profit_progress = equity / start_equity - 1.0
        if daily_dd >= cfg.daily_dd_limit:
            if position is not None:
                close_position(t, close, "DAILY_DD", row)
            stopped_reason = "daily_dd"
            break
        if total_dd >= cfg.total_dd_limit:
            if position is not None:
                close_position(t, close, "TOTAL_DD", row)
            stopped_reason = "total_dd"
            break
        if profit_progress >= cfg.profit_target:
            if position is not None:
                close_position(t, close, "PROFIT_TARGET", row)
            stopped_reason = "profit_target"
            break

        # Entry logic.
        if warmup_seen < cfg.startup_warmup_bars:
            warmup_seen += 1
            continue
        if position is not None:
            # Reverse signal closes current position, then may enter the opposite side.
            sig = str(row["signal"])
            if (position.direction == "BUY" and sig == "SELL") or (position.direction == "SELL" and sig == "BUY"):
                close_position(t, close, "REVERSE", row)
            else:
                continue

        if position is None:
            if trades_today >= cfg.max_trades_per_day or consecutive_losses >= cfg.max_consecutive_losses:
                continue
            if loss_cooldown_until is not None and t < loss_cooldown_until:
                continue
            if last_half_close_time is not None and (t - last_half_close_time).total_seconds() < cfg.half_close_cooldown_bars * 300:
                continue
            if last_trade_time is not None and (t - last_trade_time).total_seconds() < cfg.cooldown_bars_after_trade * 10:
                continue
            sig = str(row["signal"])
            if sig in {"BUY", "SELL"}:
                sl, tp = atr_sl_tp(sig, close, atr, cfg)
                risk_per_lot = abs(close - sl) * CONTRACT_SIZE
                risk_pct = effective_risk_pct(week_start, t, cfg, reset_warmup)
                volume = round_volume(min(cfg.max_lots, (equity * risk_pct) / max(risk_per_lot, 1e-9)))
                if volume >= 0.01:
                    position = Position(direction=sig, entry_time=t, entry=close, sl=sl, tp=tp, volume=volume)
                    last_trade_time = t
                    trades_today += 1

    if not week_df.empty and position is not None:
        last = week_df.iloc[-1]
        close_position(pd.Timestamp(last["time"]), float(last["close"]), "WEEK_END", last)

    end_equity = balance
    ret = end_equity / start_equity - 1.0
    wins = sum(1 for tr in trades if float(tr["pnl"]) > 0)
    losses = sum(1 for tr in trades if float(tr["pnl"]) < 0)
    gross_win = sum(float(tr["pnl"]) for tr in trades if float(tr["pnl"]) > 0)
    gross_loss = -sum(float(tr["pnl"]) for tr in trades if float(tr["pnl"]) < 0)
    return (
        {
            "week": str(week_df["week"].iloc[0]) if not week_df.empty else "",
            "start": pd.Timestamp(week_df["time"].iloc[0]).isoformat() if not week_df.empty else "",
            "end": pd.Timestamp(week_df["time"].iloc[-1]).isoformat() if not week_df.empty else "",
            "bars": int(len(week_df)),
            "return_pct": round(ret * 100.0, 4),
            "pnl": round(end_equity - start_equity, 2),
            "end_equity": round(end_equity, 2),
            "trades": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round((wins / len(trades) * 100.0) if trades else 0.0, 2),
            "profit_factor": round((gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0), 3),
            "stopped_reason": stopped_reason,
        },
        trades,
    )


def summarize(weekly: List[Dict[str, object]], trades: List[Dict[str, object]], cfg: Config, data_start: str, data_end: str, variant: str) -> Dict[str, object]:
    returns = [float(r["return_pct"]) for r in weekly]
    pnls = [float(r["pnl"]) for r in weekly]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    avg = statistics.fmean(returns) if returns else 0.0
    std = statistics.pstdev(returns) if len(returns) > 1 else 0.0
    sorted_ret = sorted(returns)
    def pct(q: float) -> float:
        if not sorted_ret:
            return 0.0
        idx = min(len(sorted_ret) - 1, max(0, int(round((len(sorted_ret) - 1) * q))))
        return sorted_ret[idx]

    total_gross_win = sum(float(t["pnl"]) for t in trades if float(t["pnl"]) > 0)
    total_gross_loss = -sum(float(t["pnl"]) for t in trades if float(t["pnl"]) < 0)
    return {
        "variant": variant,
        "symbol": SYMBOL,
        "data_start": data_start,
        "data_end": data_end,
        "weeks": len(weekly),
        "active_weeks": sum(1 for r in weekly if int(r["trades"]) > 0),
        "weekly_win_rate_pct": round(len(wins) / len(returns) * 100.0, 2) if returns else 0.0,
        "avg_week_return_pct": round(avg, 4),
        "median_week_return_pct": round(statistics.median(returns), 4) if returns else 0.0,
        "std_week_return_pct": round(std, 4),
        "sharpe_weekly_annualized": round((avg / std) * math.sqrt(52), 3) if std > 0 else 0.0,
        "best_week_pct": round(max(returns), 4) if returns else 0.0,
        "worst_week_pct": round(min(returns), 4) if returns else 0.0,
        "p05_week_pct": round(pct(0.05), 4),
        "p25_week_pct": round(pct(0.25), 4),
        "p75_week_pct": round(pct(0.75), 4),
        "p95_week_pct": round(pct(0.95), 4),
        "avg_week_pnl": round(statistics.fmean(pnls), 2) if pnls else 0.0,
        "sum_independent_week_pnl": round(sum(pnls), 2),
        "trades": len(trades),
        "avg_trades_per_week": round(len(trades) / len(weekly), 2) if weekly else 0.0,
        "trade_win_rate_pct": round(sum(1 for t in trades if float(t["pnl"]) > 0) / len(trades) * 100.0, 2) if trades else 0.0,
        "trade_profit_factor": round(total_gross_win / total_gross_loss, 3) if total_gross_loss > 0 else 999.0,
        "daily_dd_stops": sum(1 for r in weekly if r["stopped_reason"] == "daily_dd"),
        "total_dd_stops": sum(1 for r in weekly if r["stopped_reason"] == "total_dd"),
        "profit_target_stops": sum(1 for r in weekly if r["stopped_reason"] == "profit_target"),
        "config": dataclasses.asdict(cfg),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_variant(df: pd.DataFrame, cfg: Config, out_dir: Path, variant: str, reset_warmup: bool) -> Dict[str, object]:
    weekly_rows: List[Dict[str, object]] = []
    trade_rows: List[Dict[str, object]] = []
    for _, week_df in df.groupby("week", sort=True):
        if len(week_df) < 50:
            continue
        row, trades = simulate_week(week_df.reset_index(drop=True), cfg, reset_warmup=reset_warmup)
        weekly_rows.append(row)
        trade_rows.extend(trades)
    summary = summarize(
        weekly_rows,
        trade_rows,
        cfg,
        pd.Timestamp(df["time"].min()).isoformat(),
        pd.Timestamp(df["time"].max()).isoformat(),
        variant,
    )
    write_csv(out_dir / f"{variant}_weekly_returns.csv", weekly_rows)
    write_csv(out_dir / f"{variant}_trades.csv", trade_rows)
    (out_dir / f"{variant}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--refresh-data", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    end = dt.datetime.now()
    start = end - dt.timedelta(days=365 * args.years + 10)  # warmup cushion
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_ROOT / f"{args.symbol}_M5_{args.years}y_mt5.csv"
    if args.refresh_data and cache_path.exists():
        cache_path.unlink()

    raw = fetch_m5_from_mt5(args.symbol, start, end, cache_path)
    raw = raw[(raw["time"] >= pd.Timestamp(end - dt.timedelta(days=365 * args.years))) & (raw["time"] <= pd.Timestamp(end))].copy()
    if raw.empty:
        raise RuntimeError("No bars after 5-year range filter")
    df = add_indicators(raw, cfg).dropna(subset=["atr", "fast_sma", "slow_sma", "htf_fast_sma", "htf_slow_sma"]).copy()
    iso = df["time"].dt.isocalendar()
    df["week"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)

    out_dir = OUT_ROOT / now_stamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out_dir / "input_xauusd_m5.csv", index=False)
    df.to_csv(out_dir / "input_xauusd_m5_with_indicators.csv", index=False)

    summaries = [
        run_variant(df, cfg, out_dir, "active_weekly_warmup_reset", reset_warmup=True),
        run_variant(df, cfg, out_dir, "steady_state_no_warmup", reset_warmup=False),
    ]
    (out_dir / "summary_all.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "summaries": summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
