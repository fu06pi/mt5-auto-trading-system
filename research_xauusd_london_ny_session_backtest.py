#!/usr/bin/env python3
"""Read-only XAUUSD session-window backtest.

Purpose: test a non-live variant that changes only entry attention/trading windows to
London pre/open and New York pre/open windows. It reads active_plan.json for the live
parameters but does not modify active_plan.json, strategy state, MT5 orders, or live
processes.

Default test windows are UTC-based for current DST market hours:
- London pre/open/intraday: 06:00-10:00 UTC
- New York pre/open/intraday: 12:00-16:00 UTC
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path("/home/chain4655/Documents/Projects/MT5")
ACTIVE_PLAN = ROOT / "auto_quant/active_plan.json"
DATA_CACHE = Path("/home/chain4655/Documents/backtest_reports/xauusd_5y_weekly_reset/data/XAUUSD_M5_5y_mt5.csv")
OUT_ROOT = Path("/home/chain4655/Documents/backtest_reports/xauusd_london_ny_session")

SYMBOL = "XAUUSD"
POINT = 0.01
CONTRACT_SIZE = 100.0


def arg_value(cmd: Sequence[str], name: str, default: str) -> str:
    try:
        idx = list(cmd).index(name)
        return str(cmd[idx + 1])
    except (ValueError, IndexError):
        return default


def has_flag(cmd: Sequence[str], name: str) -> bool:
    return name in set(cmd)


@dataclasses.dataclass(frozen=True)
class Config:
    start_equity: float = 100000.0
    risk_pct: float = 0.0035
    daily_dd_limit: float = 0.030
    total_dd_limit: float = 0.045
    profit_target: float = 0.100
    max_lots: float = 1.8
    max_lots_per_order: float = 0.5
    fast_sma: int = 20
    slow_sma: int = 60
    htf_fast_sma: int = 50
    htf_slow_sma: int = 200
    trend_threshold: float = 0.35
    htf_comp_momentum_threshold: float = 1.10
    htf_momentum_bias_weight: float = 0.0
    momentum_score_weight: float = 0.25
    atr_period: int = 14
    breakout_lookback: int = 20
    stop_atr: float = 2.5
    reward_multiple: float = 2.5
    primary_tp_reward_multiple: float = 0.0
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
    max_hold_minutes: int = 0
    half_close_profit_usd: float = 2000.0
    half_close_fraction: float = 0.5
    half_close_cooldown_bars: int = 24
    cost_per_lot_roundtrip: float = 40.0


def load_config() -> Config:
    plan = json.loads(ACTIVE_PLAN.read_text(encoding="utf-8"))
    cmd = list(plan.get("cmd") or [])
    return Config(
        start_equity=float(arg_value(cmd, "--start-equity", "100000")),
        risk_pct=float(arg_value(cmd, "--risk-pct", "0.0035")),
        daily_dd_limit=float(arg_value(cmd, "--daily-dd-limit", "0.030")),
        total_dd_limit=float(arg_value(cmd, "--total-dd-limit", "0.045")),
        profit_target=float(arg_value(cmd, "--profit-target", "0.100")),
        max_lots=float(arg_value(cmd, "--max-lots", "1.8")),
        max_lots_per_order=float(arg_value(cmd, "--max-lots-per-order", "0.5")),
        fast_sma=int(arg_value(cmd, "--fast-sma", "20")),
        slow_sma=int(arg_value(cmd, "--slow-sma", "60")),
        htf_fast_sma=int(arg_value(cmd, "--htf-fast-sma", "50")),
        htf_slow_sma=int(arg_value(cmd, "--htf-slow-sma", "200")),
        trend_threshold=float(arg_value(cmd, "--trend-threshold", "0.35")),
        htf_comp_momentum_threshold=float(arg_value(cmd, "--htf-comp-momentum-threshold", "1.10")),
        htf_momentum_bias_weight=float(arg_value(cmd, "--htf-momentum-bias-weight", "0.0")),
        momentum_score_weight=float(arg_value(cmd, "--momentum-score-weight", "0.25")),
        atr_period=int(arg_value(cmd, "--atr-period", "14")),
        breakout_lookback=int(arg_value(cmd, "--breakout-lookback", "20")),
        stop_atr=float(arg_value(cmd, "--stop-atr", "2.5")),
        reward_multiple=float(arg_value(cmd, "--reward-multiple", "2.5")),
        primary_tp_reward_multiple=float(arg_value(cmd, "--primary-tp-reward-multiple", "0")),
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
        max_hold_minutes=int(arg_value(cmd, "--max-hold-minutes", "0")),
        half_close_profit_usd=float(arg_value(cmd, "--auto-half-profit-usd", "2000")),
        half_close_fraction=float(arg_value(cmd, "--auto-half-fraction", "0.5")),
        half_close_cooldown_bars=int(arg_value(cmd, "--half-close-cooldown-bars", "24")),
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


def clamp(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return s.clip(lower=lo, upper=hi)


def round_volume(v: float) -> float:
    return max(0.01, math.floor(v / 0.01 + 1e-9) * 0.01)


def parse_windows(text: str) -> List[Tuple[int, int]]:
    windows: List[Tuple[int, int]] = []
    for part in text.split(","):
        if not part.strip():
            continue
        start_s, end_s = part.split("-")
        start = int(start_s)
        end = int(end_s)
        if not (0 <= start <= 23 and 0 <= end <= 24):
            raise ValueError(f"bad window: {part}")
        windows.append((start, end))
    return windows


def in_windows(t: pd.Timestamp, windows: Sequence[Tuple[int, int]]) -> bool:
    if not windows:
        return True
    hour = int(t.hour)
    for start, end in windows:
        if start == end:
            return True
        if start < end and start <= hour < end:
            return True
        if start > end and (hour >= start or hour < end):
            return True
    return False


def add_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy().sort_values("time").drop_duplicates("time").reset_index(drop=True)
    out["time"] = pd.to_datetime(out["time"]).astype("datetime64[ns]")
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(cfg.atr_period).mean().clip(lower=POINT * 5)
    out["fast_sma"] = close.rolling(cfg.fast_sma).mean()
    out["slow_sma"] = close.rolling(cfg.slow_sma).mean()
    out["momentum"] = ((close - close.shift(3)) / out["atr"]).clip(-2.0, 2.0)
    lookback = max(10, cfg.breakout_lookback)
    out["recent_high"] = high.rolling(lookback).max()
    out["recent_low"] = low.rolling(lookback).min()

    h1 = (
        out.set_index("time")[["open", "high", "low", "close"]]
        .resample("1h", label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
        .reset_index()
    )
    h1["htf_fast_sma"] = h1["close"].rolling(cfg.htf_fast_sma).mean()
    h1["htf_slow_sma"] = h1["close"].rolling(cfg.htf_slow_sma).mean()
    h1_close = h1["close"]
    h1["h1_recent_slope"] = h1_close - h1_close.shift(3)
    h1["htf_signal"] = "NEUTRAL"
    h1.loc[h1["htf_fast_sma"] > h1["htf_slow_sma"], "htf_signal"] = "BULL"
    h1.loc[h1["htf_fast_sma"] < h1["htf_slow_sma"], "htf_signal"] = "BEAR"
    h1["time"] = (h1["time"] + pd.Timedelta(microseconds=1)).astype("datetime64[ns]")
    out = pd.merge_asof(
        out.sort_values("time"),
        h1[["time", "htf_fast_sma", "htf_slow_sma", "htf_signal"]].sort_values("time"),
        on="time",
        direction="backward",
    )

    m15 = (
        out.set_index("time")[["open", "high", "low", "close"]]
        .resample("15min", label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
        .reset_index()
    )
    m15["m15_close_m3"] = m15["close"].shift(3)
    m15["time"] = (m15["time"] + pd.Timedelta(microseconds=1)).astype("datetime64[ns]")
    out = pd.merge_asof(
        out.sort_values("time"),
        m15[["time", "close", "m15_close_m3"]].rename(columns={"close": "m15_close"}).sort_values("time"),
        on="time",
        direction="backward",
    )
    out["m15_momentum"] = ((out["m15_close"] - out["m15_close_m3"]) / out["atr"]).clip(-2.0, 2.0).fillna(0.0)
    comp_th = max(cfg.htf_comp_momentum_threshold, cfg.trend_threshold * 2.0)
    out["htf_comp"] = out["htf_signal"]
    out.loc[out["m15_momentum"] >= comp_th, "htf_comp"] = "BULL"
    out.loc[out["m15_momentum"] <= -comp_th, "htf_comp"] = "BEAR"

    trend = pd.Series(0.0, index=out.index)
    trend[(out["close"] > out["fast_sma"]) & (out["fast_sma"] > out["slow_sma"])] = 0.55
    trend[(out["close"] < out["fast_sma"]) & (out["fast_sma"] < out["slow_sma"])] = -0.55
    trend[(trend == 0.0) & (out["close"] > out["slow_sma"])] = 0.20
    trend[(trend == 0.0) & (out["close"] < out["slow_sma"])] = -0.20

    htf_gap = out["htf_fast_sma"] - out["htf_slow_sma"]
    htf_gap_strength = (htf_gap / out["atr"].clip(lower=POINT * 5)).clip(-1.5, 1.5)
    htf_momentum = out["momentum"].clip(-2.0, 2.0)
    htf_bias = htf_gap_strength * 0.28 + htf_momentum * cfg.htf_momentum_bias_weight
    bull_bias = 0.22 + htf_gap_strength.clip(lower=0.0) * 0.18
    bear_bias = -0.22 + htf_gap_strength.clip(upper=0.0) * 0.18
    htf_bias = htf_bias.where(out["htf_comp"] != "BULL", pd.concat([htf_bias, bull_bias], axis=1).max(axis=1))
    htf_bias = htf_bias.where(out["htf_comp"] != "BEAR", pd.concat([htf_bias, bear_bias], axis=1).min(axis=1))
    htf_bias = htf_bias.where(out["htf_comp"] != "NEUTRAL", 0.0)
    htf_bias = htf_bias.where(~((out["htf_comp"] == "BULL") & (htf_momentum < 0)), htf_bias * 0.65)
    htf_bias = htf_bias.where(~((out["htf_comp"] == "BEAR") & (htf_momentum > 0)), htf_bias * 0.65)

    breakout = ((out["close"] - out["recent_high"]) / out["atr"]).clip(-1, 1) * 0.30
    breakout += ((out["recent_low"] - out["close"]) / out["atr"]).clip(-1, 1) * -0.30
    momentum_component = out["momentum"].clip(-1.5, 1.5) * cfg.momentum_score_weight
    out["score"] = (trend + htf_bias + breakout + momentum_component).clip(-1.5, 1.5)
    out["signal"] = "NONE"
    out.loc[(out["htf_comp"] == "BULL") & (out["score"] >= cfg.trend_threshold), "signal"] = "BUY"
    out.loc[(out["htf_comp"] == "BEAR") & (out["score"] <= -cfg.trend_threshold), "signal"] = "SELL"
    return out.dropna(subset=["atr", "fast_sma", "slow_sma", "htf_fast_sma", "htf_slow_sma"]).reset_index(drop=True)


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
        favorable = close - pos.entry
        if favorable >= cfg.break_even_atr * atr:
            pos.sl = max(pos.sl, round(pos.entry + cfg.break_even_lock_atr * atr, 2))
        if favorable >= cfg.trail_trigger_atr * atr:
            pos.sl = max(pos.sl, round(close - cfg.trail_lock_atr * atr, 2))
    else:
        favorable = pos.entry - close
        if favorable >= cfg.break_even_atr * atr:
            pos.sl = min(pos.sl, round(pos.entry - cfg.break_even_lock_atr * atr, 2))
        if favorable >= cfg.trail_trigger_atr * atr:
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


def simulate(df: pd.DataFrame, cfg: Config, windows: Sequence[Tuple[int, int]]) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    balance = cfg.start_equity
    day_start_equity = cfg.start_equity
    max_equity = cfg.start_equity
    current_day: Optional[dt.date] = None
    trades_today = 0
    position: Optional[Position] = None
    consecutive_losses = 0
    loss_cooldown_until: Optional[pd.Timestamp] = None
    last_trade_time: Optional[pd.Timestamp] = None
    last_half_close_time: Optional[pd.Timestamp] = None
    warmup_seen = 0
    stopped_reason = ""
    trades: List[Dict[str, object]] = []
    equity_curve: List[Dict[str, object]] = []

    def equity(price: float) -> float:
        return balance + (floating_pnl(position, price) if position else 0.0)

    def close_position(t: pd.Timestamp, price: float, reason: str, row: pd.Series, volume_override: Optional[float] = None) -> None:
        nonlocal balance, position, consecutive_losses, loss_cooldown_until, last_half_close_time
        if position is None:
            return
        volume = position.volume if volume_override is None else min(position.volume, volume_override)
        pnl = pnl_for(position.direction, position.entry, price, volume, cfg.cost_per_lot_roundtrip)
        balance += pnl
        trades.append({
            "entry_time": position.entry_time.isoformat(),
            "exit_time": t.isoformat(),
            "direction": position.direction,
            "entry": round(position.entry, 2),
            "exit": round(price, 2),
            "volume": round(volume, 2),
            "pnl": round(pnl, 2),
            "reason": reason,
            "entry_hour_utc": int(position.entry_time.hour),
            "exit_hour_utc": int(t.hour),
        })
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

    for _, row in df.iterrows():
        t = pd.Timestamp(row["time"])
        close = float(row["close"])
        atr = float(row["atr"])
        if current_day != t.date():
            current_day = t.date()
            day_start_equity = equity(close)
            trades_today = 0
        if position is not None:
            update_trail(position, close, atr, cfg)
            reason, exit_price = exit_hit(position, row)
            if reason and exit_price is not None:
                close_position(t, float(exit_price), reason, row)
            elif position is not None and cfg.max_hold_minutes > 0 and (t - position.entry_time).total_seconds() / 60.0 >= cfg.max_hold_minutes:
                close_position(t, close, "MAX_HOLD", row)
            elif position is not None and floating_pnl(position, close) >= cfg.half_close_profit_usd and not position.half_done:
                half_volume = round_volume(position.volume * cfg.half_close_fraction)
                if 0.01 <= half_volume < position.volume:
                    close_position(t, close, "HALF_PROFIT", row, volume_override=half_volume)

        eq = equity(close)
        max_equity = max(max_equity, eq)
        daily_dd = 1.0 - eq / max(day_start_equity, 1e-9)
        total_dd = 1.0 - eq / max(max_equity, 1e-9)
        equity_curve.append({"time": t.isoformat(), "equity": round(eq, 2), "balance": round(balance, 2), "total_dd_pct": round(total_dd * 100, 4)})
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
        if eq / cfg.start_equity - 1.0 >= cfg.profit_target:
            if position is not None:
                close_position(t, close, "PROFIT_TARGET", row)
            stopped_reason = "profit_target"
            break

        if warmup_seen < cfg.startup_warmup_bars:
            warmup_seen += 1
            continue
        if position is not None:
            sig = str(row["signal"])
            if (position.direction == "BUY" and sig == "SELL") or (position.direction == "SELL" and sig == "BUY"):
                close_position(t, close, "REVERSE", row)
            else:
                continue
        if position is None:
            if not in_windows(t, windows):
                continue
            if trades_today >= cfg.max_trades_per_day or consecutive_losses >= cfg.max_consecutive_losses:
                continue
            if loss_cooldown_until is not None and t < loss_cooldown_until:
                continue
            if last_half_close_time is not None and (t - last_half_close_time).total_seconds() < cfg.half_close_cooldown_bars * 300:
                continue
            if last_trade_time is not None and (t - last_trade_time).total_seconds() < cfg.cooldown_bars_after_trade * 300:
                continue
            sig = str(row["signal"])
            if sig in {"BUY", "SELL"}:
                sl, tp = atr_sl_tp(sig, close, atr, cfg)
                risk_per_lot = abs(close - sl) * CONTRACT_SIZE
                lots_by_risk = (eq * cfg.risk_pct) / max(risk_per_lot, 1e-9)
                volume = round_volume(min(cfg.max_lots, cfg.max_lots_per_order, lots_by_risk))
                if volume >= 0.01:
                    position = Position(sig, t, close, sl, tp, volume)
                    last_trade_time = t
                    trades_today += 1

    if not df.empty and position is not None:
        last = df.iloc[-1]
        close_position(pd.Timestamp(last["time"]), float(last["close"]), "END", last)

    end_equity = balance
    gross_win = sum(float(t["pnl"]) for t in trades if float(t["pnl"]) > 0)
    gross_loss = -sum(float(t["pnl"]) for t in trades if float(t["pnl"]) < 0)
    max_dd = max((float(x["total_dd_pct"]) for x in equity_curve), default=0.0)
    wins = sum(1 for t in trades if float(t["pnl"]) > 0)
    summary = {
        "symbol": SYMBOL,
        "variant": "london_ny_preopen_intraday_windows",
        "windows_utc": [f"{a:02d}-{b:02d}" for a, b in windows],
        "data_start": pd.Timestamp(df["time"].min()).isoformat() if not df.empty else "",
        "data_end": pd.Timestamp(df["time"].max()).isoformat() if not df.empty else "",
        "bars": int(len(df)),
        "final_equity": round(end_equity, 2),
        "return_pct": round((end_equity / cfg.start_equity - 1.0) * 100.0, 4),
        "max_dd_pct": round(max_dd, 4),
        "trades": len(trades),
        "win_rate_pct": round((wins / len(trades) * 100.0) if trades else 0.0, 2),
        "profit_factor": round((gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0), 3),
        "gross_win": round(gross_win, 2),
        "gross_loss": round(gross_loss, 2),
        "stopped_reason": stopped_reason,
        "config": dataclasses.asdict(cfg),
    }
    return summary, trades, equity_curve


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_CACHE))
    parser.add_argument("--windows-utc", default="06-10,12-16", help="Comma windows, e.g. 06-10,12-16. Empty = all day.")
    parser.add_argument("--years", type=float, default=5.0)
    args = parser.parse_args()

    cfg = load_config()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"missing broker M5 cache: {data_path}")
    raw = pd.read_csv(data_path, parse_dates=["time"])
    raw = raw.sort_values("time").drop_duplicates("time")
    if args.years > 0:
        end = pd.Timestamp(raw["time"].max())
        raw = raw[raw["time"] >= end - pd.Timedelta(days=365 * args.years)].copy()
    df = add_indicators(raw, cfg)
    windows = parse_windows(args.windows_utc) if args.windows_utc.strip() else []
    summary, trades, equity_curve = simulate(df, cfg, windows)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out_dir / "trades.csv", trades)
    write_csv(out_dir / "equity_curve.csv", equity_curve)
    print(json.dumps({"out_dir": str(out_dir), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
