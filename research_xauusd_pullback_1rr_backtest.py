#!/usr/bin/env python3
"""Research-only backtest for current XAUUSD trend pullback variants.

This script reuses the existing weekly-reset simulator and cached broker M5 data.
It tests:
- option 1: pullback entries with RR 1.5 / 2.0 / 2.5.
- option 3: pullback plus confirmation breakout through the previous bar high/low.
- option 4: high-ADX early-trend entries.

It does not read or modify live state, and it does not touch active_plan.json.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

import research_xauusd_5y_weekly_reset_backtest as base

ROOT = Path("/home/chain4655/Documents/Projects/MT5")
OUT_ROOT = Path("/home/chain4655/Documents/backtest_reports/xauusd_pullback_1rr")
TEST_PLAN = ROOT / "auto_quant/test_plans/xauusd_trend_pullback_1rr_plan.json"
ACTIVE_PLAN = ROOT / "auto_quant/active_plan.json"


def arg_value(cmd: Sequence[str], name: str, default: str) -> str:
    try:
        idx = list(cmd).index(name)
        return str(cmd[idx + 1])
    except (ValueError, IndexError):
        return default


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100.0 * plus_dm.rolling(period).mean() / atr.replace(0, math.nan)
    minus_di = 100.0 * minus_dm.rolling(period).mean() / atr.replace(0, math.nan)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, math.nan)) * 100.0
    return dx.rolling(period).mean().fillna(0.0)


def add_trend_stage_columns(out: pd.DataFrame) -> pd.DataFrame:
    """Add closed-bar trend age and ADX acceleration diagnostics.

    trend_age_h1 approximates how old the raw H1 HTF regime is.  A value near
    zero means the H1 fast/slow SMA regime just flipped; values above 24 mean
    the trend is already more than about one trading day old.
    """
    regime = out["htf_signal"].where(out["htf_signal"].isin(["BULL", "BEAR"]), "NEUTRAL")
    groups = regime.ne(regime.shift(1)).cumsum()
    out["trend_age_m5_bars"] = out.groupby(groups).cumcount()
    out["trend_age_h1"] = out["trend_age_m5_bars"] / 12.0
    out["adx_delta_6"] = out["adx"] - out["adx"].shift(6)
    out["adx_delta_12"] = out["adx"] - out["adx"].shift(12)
    return out


def apply_quality_and_entry_gate(
    df: pd.DataFrame,
    *,
    min_abs_score: float,
    min_adx: float,
    entry_mode: str,
    pullback_max_atr: float,
    max_trend_age_h1: Optional[float] = None,
    min_trend_age_h1: float = 0.0,
    require_adx_rising: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    out["adx"] = compute_adx(out, 14)
    out = add_trend_stage_columns(out)

    signal_mask = out["signal"].isin(["BUY", "SELL"])
    if min_abs_score > 0:
        out.loc[signal_mask & (out["score"].abs() < min_abs_score), "signal"] = "NONE"
        signal_mask = out["signal"].isin(["BUY", "SELL"])
    if min_adx > 0:
        out.loc[signal_mask & (out["adx"] < min_adx), "signal"] = "NONE"
        signal_mask = out["signal"].isin(["BUY", "SELL"])
    if max_trend_age_h1 is not None:
        early_ok = (out["trend_age_h1"] >= min_trend_age_h1) & (
            out["trend_age_h1"] <= max_trend_age_h1
        )
        out.loc[signal_mask & ~early_ok, "signal"] = "NONE"
        signal_mask = out["signal"].isin(["BUY", "SELL"])
    if require_adx_rising:
        rising_ok = (out["adx_delta_6"] > 0.0) & (out["adx_delta_12"] > 0.0)
        out.loc[signal_mask & ~rising_ok, "signal"] = "NONE"

    mode = entry_mode.lower()
    if mode in {"pullback", "pullback_breakout"}:
        zone = pullback_max_atr * out["atr"]
        buy_ok = (
            (out["signal"] == "BUY")
            & (out["low"] <= out["fast_sma"] + zone)
            & (out["close"] >= out["fast_sma"])
        )
        sell_ok = (
            (out["signal"] == "SELL")
            & (out["high"] >= out["fast_sma"] - zone)
            & (out["close"] <= out["fast_sma"])
        )
        if mode == "pullback_breakout":
            # Option 3: retest is valid only if the close re-breaks the previous
            # completed bar's high/low in the trend direction.
            buy_ok = buy_ok & (out["close"] > out["high"].shift(1))
            sell_ok = sell_ok & (out["close"] < out["low"].shift(1))
        out.loc[out["signal"].isin(["BUY", "SELL"]) & ~(buy_ok | sell_ok), "signal"] = "NONE"
    return out


def run() -> Dict[str, object]:
    active_hash = hashlib.sha256(ACTIVE_PLAN.read_bytes()).hexdigest()
    test_plan = json.loads(TEST_PLAN.read_text(encoding="utf-8"))
    test_cmd: List[str] = list(test_plan["cmd"])

    base_cfg = base.load_config()
    min_abs_score = float(arg_value(test_cmd, "--min-abs-score", "0"))
    min_adx = float(arg_value(test_cmd, "--min-adx", "0"))
    pullback_max_atr = float(arg_value(test_cmd, "--pullback-max-atr", "0.35"))

    cache_path = base.DATA_ROOT / "XAUUSD_M5_5y_mt5.csv"
    raw = pd.read_csv(cache_path, parse_dates=["time"])
    raw = raw.sort_values("time").reset_index(drop=True)
    data_start = pd.Timestamp(raw["time"].min()).isoformat()
    data_end = pd.Timestamp(raw["time"].max()).isoformat()

    # Indicators are independent of RR, so build once using current-plan signal knobs.
    cfg_for_signals = dataclasses.replace(base_cfg, cost_per_lot_roundtrip=40.0)
    df0 = base.add_indicators(raw, cfg_for_signals).dropna(
        subset=["atr", "fast_sma", "slow_sma", "htf_fast_sma", "htf_slow_sma"]
    ).copy()
    iso = df0["time"].dt.isocalendar()
    df0["week"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)

    out_dir = OUT_ROOT / now_stamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "source": "cached broker M5 CSV",
        "cache_path": str(cache_path),
        "data_start": data_start,
        "data_end": data_end,
        "active_plan_sha256_before": active_hash,
        "test_plan": str(TEST_PLAN),
        "min_abs_score": min_abs_score,
        "min_adx": min_adx,
        "pullback_max_atr": pullback_max_atr,
        "notes": [
            "Research-only simulator; active_plan.json is not modified.",
            "Uses existing weekly-reset simulator; live-only mechanics are approximated.",
            "Option 1: pullback RR sweep 1.5/2.0/2.5.",
            "Option 3: pullback plus close through previous bar high/low confirmation.",
            "Option 4: high-ADX early-trend gate uses raw H1 regime age and rising ADX diagnostics.",
        ],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    summaries = []
    variants = []
    for rr in [1.0, 1.5, 2.0, 2.5]:
        variants.append({
            "variant": f"pullback_rr{str(rr).replace('.', 'p')}",
            "mode": "pullback",
            "rr": rr,
            "min_adx_override": min_adx,
        })
    for rr in [1.0, 1.5, 2.0, 2.5]:
        variants.append({
            "variant": f"pullback_breakout_rr{str(rr).replace('.', 'p')}",
            "mode": "pullback_breakout",
            "rr": rr,
            "min_adx_override": min_adx,
        })
    # Option 4: high ADX + early raw-H1 trend stage.  Test both immediate
    # entries and pullback-gated entries; RR is kept at 1.0 because the earlier
    # sweep showed higher RR rarely reaches TP under current exits/costs.
    for adx_threshold in [28.0, 32.0, 36.0]:
        for max_age_h1 in [6.0, 12.0, 24.0]:
            variants.append({
                "variant": f"early_adx{int(adx_threshold)}_age{int(max_age_h1)}h_rr1p0",
                "mode": "immediate",
                "rr": 1.0,
                "min_adx_override": adx_threshold,
                "max_trend_age_h1": max_age_h1,
                "require_adx_rising": True,
            })
            variants.append({
                "variant": f"early_adx{int(adx_threshold)}_age{int(max_age_h1)}h_pullback_rr1p0",
                "mode": "pullback",
                "rr": 1.0,
                "min_adx_override": adx_threshold,
                "max_trend_age_h1": max_age_h1,
                "require_adx_rising": True,
            })

    (out_dir / "variants_spec.json").write_text(
        json.dumps(variants, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for item in variants:
        variant = str(item["variant"])
        mode = str(item["mode"])
        rr = float(item["rr"])
        cfg = dataclasses.replace(
            base_cfg,
            reward_multiple=rr,
            primary_tp_reward_multiple=rr,
            cost_per_lot_roundtrip=40.0,
            # Live plan has --auto-half-profit-usd 0, meaning disabled. The shared
            # research simulator treats 0 as an immediate threshold, so use a
            # practically unreachable value to preserve live-disabled behavior.
            half_close_profit_usd=1_000_000_000.0,
        )
        df = apply_quality_and_entry_gate(
            df0,
            min_abs_score=min_abs_score,
            min_adx=float(item.get("min_adx_override", min_adx)),
            entry_mode=mode,
            pullback_max_atr=pullback_max_atr,
            max_trend_age_h1=item.get("max_trend_age_h1"),
            require_adx_rising=bool(item.get("require_adx_rising", False)),
        )
        df.to_csv(out_dir / f"{variant}_signals.csv", index=False)
        summaries.append(base.run_variant(df, cfg, out_dir, variant, reset_warmup=False))

    result = {"out_dir": str(out_dir), "meta": meta, "summaries": summaries}
    (out_dir / "summary_all.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
