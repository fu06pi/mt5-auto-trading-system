#!/usr/bin/env python3
"""Research-only XAUUSD FVG/OB/liquidity-overlap zone retest backtest.

Scope / design (minimal first pass):
- Input: cached broker M5 OHLCV CSV, defaulting to the existing XAUUSD cache under
  /home/chain4655/Documents/backtest_reports/xauusd_5y_weekly_reset/data/.
- Read-only: does not import or modify live active_plan / MT5 bridge state.
- Zone memory: objective 3-candle FVG zones are stored until max age, mitigation,
  or invalidation.
- Confluence proxies:
  * OB overlap: recent opposite candle body/range overlaps the FVG zone.
  * Liquidity overlap: recent swing sweep/reclaim overlaps directional intent.
  * Optional H1 EMA trend context is aggregated from the same M5 cache.
- Entry: first retest touch of the live zone with close reclaim in zone direction.
- Risk model: fixed 0.01 lot, conservative SL-first fill when SL/TP both hit.

This is intentionally a compact research scaffold, not live trading code.
Outputs are written to /home/chain4655/Documents/backtest_reports/zone_retest_research/.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

DEFAULT_DATA = Path("/home/chain4655/Documents/backtest_reports/xauusd_5y_weekly_reset/data/XAUUSD_M5_5y_mt5.csv")
DEFAULT_OUT = Path("/home/chain4655/Documents/backtest_reports/zone_retest_research")

INITIAL_EQUITY = 10_000.0
CONTRACT_SIZE = 100.0
FIXED_LOTS = 0.01
SPREAD_PRICE = 0.39
COMMISSION_PER_LOT_ROUND = 7.0
POINT = 0.01


@dataclasses.dataclass(frozen=True)
class Params:
    name: str = "fvg_ob_liq_retest_v0"
    atr_period: int = 14
    min_gap_atr: float = 0.08
    max_zone_age_bars: int = 288  # one M5 day
    min_retest_age_bars: int = 2
    ob_lookback: int = 12
    liquidity_lookback: int = 36
    swing_lookback: int = 12
    require_ob_overlap: bool = False
    require_liquidity_overlap: bool = False
    require_h1_trend: bool = False
    rr: float = 1.5
    sl_atr_buffer: float = 0.35
    max_hold_bars: int = 144
    max_trades_day: int = 6
    start: Optional[str] = None
    end: Optional[str] = None


@dataclasses.dataclass
class Zone:
    zone_id: int
    created_i: int
    created_time: pd.Timestamp
    direction: int  # 1 long / -1 short
    bottom: float
    top: float
    gap_atr: float
    ob_overlap: bool
    liquidity_overlap: bool
    h1_trend: str
    touches: int = 0
    status: str = "active"


@dataclasses.dataclass
class Position:
    zone: Zone
    entry_i: int
    entry_time: pd.Timestamp
    entry: float
    sl: float
    tp: float


def load_data(path: Path, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    need = {"time", "open", "high", "low", "close"}
    missing = need.difference(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    if "tick_volume" not in df.columns:
        df["tick_volume"] = 0.0
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if start:
        df = df[df["time"] >= pd.Timestamp(start)]
    if end:
        df = df[df["time"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)[["time", "open", "high", "low", "close", "tick_volume"]]


def add_atr(df: pd.DataFrame, period: int) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr"] = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return out


def add_h1_trend(df: pd.DataFrame) -> pd.DataFrame:
    h1 = (
        df.set_index("time")
        .resample("60min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "tick_volume": "sum"})
        .dropna()
    )
    h1["ema_fast"] = h1["close"].ewm(span=20, adjust=False).mean()
    h1["ema_slow"] = h1["close"].ewm(span=50, adjust=False).mean()
    h1["h1_trend"] = "NEUTRAL"
    h1.loc[h1["ema_fast"] > h1["ema_slow"], "h1_trend"] = "UP"
    h1.loc[h1["ema_fast"] < h1["ema_slow"], "h1_trend"] = "DOWN"
    mapped = pd.merge_asof(
        df.sort_values("time"),
        h1[["h1_trend"]].reset_index().sort_values("time"),
        on="time",
        direction="backward",
    )
    mapped["h1_trend"] = mapped["h1_trend"].fillna("NEUTRAL")
    return mapped


def safe_atr(df: pd.DataFrame, i: int) -> float:
    val = float(df.at[i, "atr"])
    return POINT if math.isnan(val) or val <= 0 else val


def overlaps(a_bottom: float, a_top: float, b_bottom: float, b_top: float) -> bool:
    return max(a_bottom, b_bottom) <= min(a_top, b_top)


def has_ob_overlap(df: pd.DataFrame, i: int, direction: int, bottom: float, top: float, lookback: int) -> bool:
    start = max(0, i - lookback)
    for j in range(i - 1, start - 1, -1):
        o, h, l, c = (float(df.at[j, x]) for x in ("open", "high", "low", "close"))
        rng = max(h - l, POINT)
        body_ratio = abs(c - o) / rng
        opposite = c < o if direction == 1 else c > o
        if opposite and body_ratio >= 0.45 and overlaps(bottom, top, l, h):
            return True
    return False


def has_liquidity_overlap(df: pd.DataFrame, i: int, direction: int, lookback: int, swing_lookback: int) -> bool:
    start = max(swing_lookback + 1, i - lookback)
    for j in range(i - 1, start - 1, -1):
        prior = df.iloc[j - swing_lookback : j]
        if direction == 1:
            prior_low = float(prior["low"].min())
            if float(df.at[j, "low"]) < prior_low and float(df.at[j, "close"]) > prior_low:
                return True
        else:
            prior_high = float(prior["high"].max())
            if float(df.at[j, "high"]) > prior_high and float(df.at[j, "close"]) < prior_high:
                return True
    return False


def detect_new_zone(df: pd.DataFrame, i: int, params: Params, next_id: int) -> Optional[Zone]:
    if i < max(2, params.atr_period):
        return None
    atr = safe_atr(df, i)
    hi_2, lo_2 = float(df.at[i - 2, "high"]), float(df.at[i - 2, "low"])
    hi, lo = float(df.at[i, "high"]), float(df.at[i, "low"])
    direction = 0
    bottom = top = 0.0
    gap = 0.0
    if lo > hi_2:
        direction, bottom, top, gap = 1, hi_2, lo, lo - hi_2
    elif hi < lo_2:
        direction, bottom, top, gap = -1, hi, lo_2, lo_2 - hi
    if direction == 0 or gap / atr < params.min_gap_atr:
        return None
    h1_trend = str(df.at[i, "h1_trend"])
    ob_ok = has_ob_overlap(df, i, direction, bottom, top, params.ob_lookback)
    liq_ok = has_liquidity_overlap(df, i, direction, params.liquidity_lookback, params.swing_lookback)
    return Zone(
        zone_id=next_id,
        created_i=i,
        created_time=df.at[i, "time"],
        direction=direction,
        bottom=bottom,
        top=top,
        gap_atr=gap / atr,
        ob_overlap=ob_ok,
        liquidity_overlap=liq_ok,
        h1_trend=h1_trend,
    )


def zone_is_eligible(zone: Zone, params: Params) -> bool:
    if params.require_ob_overlap and not zone.ob_overlap:
        return False
    if params.require_liquidity_overlap and not zone.liquidity_overlap:
        return False
    if params.require_h1_trend:
        if zone.direction == 1 and zone.h1_trend != "UP":
            return False
        if zone.direction == -1 and zone.h1_trend != "DOWN":
            return False
    return True


def retest_signal(df: pd.DataFrame, i: int, zone: Zone, params: Params) -> bool:
    if i - zone.created_i < params.min_retest_age_bars:
        return False
    h, l, o, c = (float(df.at[i, x]) for x in ("high", "low", "open", "close"))
    touched = l <= zone.top and h >= zone.bottom
    if not touched:
        return False
    mid = (zone.bottom + zone.top) / 2.0
    if zone.direction == 1:
        return c > o and c >= mid
    return c < o and c <= mid


def pnl_usd(direction: int, entry: float, exit_price: float) -> float:
    gross = (exit_price - entry) * direction * CONTRACT_SIZE * FIXED_LOTS
    cost = SPREAD_PRICE * CONTRACT_SIZE * FIXED_LOTS + COMMISSION_PER_LOT_ROUND * FIXED_LOTS
    return gross - cost


def backtest(df: pd.DataFrame, params: Params) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
    equity = INITIAL_EQUITY
    peak = equity
    max_dd = 0.0
    zones: List[Zone] = []
    pos: Optional[Position] = None
    trades: List[Dict[str, Any]] = []
    trades_day: Dict[str, int] = {}
    next_zone_id = 1
    counters: Dict[str, int] = {
        "zones_created": 0,
        "zones_ob_overlap": 0,
        "zones_liquidity_overlap": 0,
        "zones_h1_aligned": 0,
        "zones_expired": 0,
        "zones_invalidated": 0,
        "retests_seen": 0,
        "entries_blocked_daily_or_position": 0,
    }

    for i in range(params.atr_period + 5, len(df)):
        bar_time = df.at[i, "time"]
        h, l, c = (float(df.at[i, x]) for x in ("high", "low", "close"))

        if pos is not None:
            direction = pos.zone.direction
            hit_sl = l <= pos.sl if direction == 1 else h >= pos.sl
            hit_tp = h >= pos.tp if direction == 1 else l <= pos.tp
            reason = ""
            exit_price = 0.0
            if hit_sl:
                reason, exit_price = "SL", pos.sl
            elif hit_tp:
                reason, exit_price = "TP", pos.tp
            elif i - pos.entry_i >= params.max_hold_bars:
                reason, exit_price = "TIME", c
            if reason:
                profit = pnl_usd(direction, pos.entry, exit_price)
                equity += profit
                peak = max(peak, equity)
                max_dd = max(max_dd, (peak - equity) / peak if peak else 0.0)
                trades.append(
                    {
                        "entry_time": pos.entry_time,
                        "exit_time": bar_time,
                        "side": "BUY" if direction == 1 else "SELL",
                        "entry": pos.entry,
                        "sl": pos.sl,
                        "tp": pos.tp,
                        "exit": exit_price,
                        "reason": reason,
                        "pnl": profit,
                        "equity": equity,
                        "bars_held": i - pos.entry_i,
                        "zone_id": pos.zone.zone_id,
                        "zone_bottom": pos.zone.bottom,
                        "zone_top": pos.zone.top,
                        "gap_atr": pos.zone.gap_atr,
                        "ob_overlap": pos.zone.ob_overlap,
                        "liquidity_overlap": pos.zone.liquidity_overlap,
                        "h1_trend": pos.zone.h1_trend,
                    }
                )
                pos.zone.status = "traded"
                pos = None

        # expire / invalidate zones before considering fresh retest entries
        active: List[Zone] = []
        for zone in zones:
            if zone.status != "active":
                continue
            if i - zone.created_i > params.max_zone_age_bars:
                zone.status = "expired"
                counters["zones_expired"] += 1
                continue
            # Full close beyond the far side invalidates the zone.
            if (zone.direction == 1 and c < zone.bottom) or (zone.direction == -1 and c > zone.top):
                zone.status = "invalidated"
                counters["zones_invalidated"] += 1
                continue
            active.append(zone)
        zones = active

        if pos is None:
            day = str(pd.Timestamp(bar_time).date())
            for zone in sorted(zones, key=lambda z: z.created_i, reverse=True):
                if not zone_is_eligible(zone, params):
                    continue
                if retest_signal(df, i, zone, params):
                    counters["retests_seen"] += 1
                    zone.touches += 1
                    if trades_day.get(day, 0) >= params.max_trades_day:
                        counters["entries_blocked_daily_or_position"] += 1
                        break
                    atr = safe_atr(df, i)
                    entry = c
                    if zone.direction == 1:
                        sl = min(zone.bottom, l) - atr * params.sl_atr_buffer
                        risk = max(entry - sl, POINT)
                        tp = entry + risk * params.rr
                    else:
                        sl = max(zone.top, h) + atr * params.sl_atr_buffer
                        risk = max(sl - entry, POINT)
                        tp = entry - risk * params.rr
                    pos = Position(zone=zone, entry_i=i, entry_time=bar_time, entry=entry, sl=sl, tp=tp)
                    trades_day[day] = trades_day.get(day, 0) + 1
                    break

        new_zone = detect_new_zone(df, i, params, next_zone_id)
        if new_zone is not None:
            zones.append(new_zone)
            next_zone_id += 1
            counters["zones_created"] += 1
            counters["zones_ob_overlap"] += int(new_zone.ob_overlap)
            counters["zones_liquidity_overlap"] += int(new_zone.liquidity_overlap)
            counters["zones_h1_aligned"] += int((new_zone.direction == 1 and new_zone.h1_trend == "UP") or (new_zone.direction == -1 and new_zone.h1_trend == "DOWN"))

    if pos is not None:
        c = float(df.at[len(df) - 1, "close"])
        profit = pnl_usd(pos.zone.direction, pos.entry, c)
        equity += profit
        trades.append(
            {
                "entry_time": pos.entry_time,
                "exit_time": df.at[len(df) - 1, "time"],
                "side": "BUY" if pos.zone.direction == 1 else "SELL",
                "entry": pos.entry,
                "sl": pos.sl,
                "tp": pos.tp,
                "exit": c,
                "reason": "EOD",
                "pnl": profit,
                "equity": equity,
                "bars_held": len(df) - 1 - pos.entry_i,
                "zone_id": pos.zone.zone_id,
                "zone_bottom": pos.zone.bottom,
                "zone_top": pos.zone.top,
                "gap_atr": pos.zone.gap_atr,
                "ob_overlap": pos.zone.ob_overlap,
                "liquidity_overlap": pos.zone.liquidity_overlap,
                "h1_trend": pos.zone.h1_trend,
            }
        )

    tdf = pd.DataFrame(trades)
    wins = int((tdf["pnl"] > 0).sum()) if not tdf.empty else 0
    losses = int((tdf["pnl"] <= 0).sum()) if not tdf.empty else 0
    gross_profit = float(tdf.loc[tdf["pnl"] > 0, "pnl"].sum()) if not tdf.empty else 0.0
    gross_loss = float(-tdf.loc[tdf["pnl"] <= 0, "pnl"].sum()) if not tdf.empty else 0.0
    summary = {
        "name": params.name,
        "bars": len(df),
        "start": df.at[0, "time"].isoformat() if len(df) else None,
        "end": df.at[len(df) - 1, "time"].isoformat() if len(df) else None,
        "trades": len(tdf),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / len(tdf), 4) if len(tdf) else 0.0,
        "net_pnl": round(float(tdf["pnl"].sum()) if not tdf.empty else 0.0, 2),
        "ending_equity": round(equity, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss else None,
        "avg_pnl": round(float(tdf["pnl"].mean()) if not tdf.empty else 0.0, 3),
        "rr": params.rr,
        "min_gap_atr": params.min_gap_atr,
        "require_ob_overlap": params.require_ob_overlap,
        "require_liquidity_overlap": params.require_liquidity_overlap,
        "require_h1_trend": params.require_h1_trend,
    }
    return summary, tdf, counters


def write_design(out_dir: Path, data_path: Path, summary: Dict[str, Any], counters: Dict[str, Any]) -> None:
    text = f"""# XAUUSD zone retest research v0

## 可用 input
- CSV: `{data_path}`
- 欄位: `time, open, high, low, close, tick_volume`
- 本次列數: {summary.get('bars')}
- 範圍: {summary.get('start')} → {summary.get('end')}

## 最小規則設計
1. 以 M5 三根 K 偵測 FVG：bullish `low[i] > high[i-2]`，bearish `high[i] < low[i-2]`。
2. FVG 進入 zone memory；超過 max age、被遠端收盤突破即失效。
3. OB overlap 是研究 proxy：近 N 根反向實體 K 的 high/low 與 FVG zone 重疊。
4. Liquidity overlap 是研究 proxy：近 N 根發生 swing high/low 掃流動性後 reclaim。
5. H1 trend 由同一份 M5 cache resample 後 EMA20/EMA50 判定。
6. retest entry：zone 仍 active、觸碰 zone，且收盤向 zone 方向 reclaim；固定 0.01 lot、SL 在 zone 外加 ATR buffer、TP = RR。

## 本次 counters
```json
{json.dumps(counters, ensure_ascii=False, indent=2)}
```

## 本次 summary
```json
{json.dumps(summary, ensure_ascii=False, indent=2)}
```
"""
    (out_dir / "zone_retest_design.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Research-only XAUUSD FVG/OB/liquidity zone retest backtest")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--require-ob-overlap", action="store_true")
    p.add_argument("--require-liquidity-overlap", action="store_true")
    p.add_argument("--require-h1-trend", action="store_true")
    p.add_argument("--rr", type=float, default=1.5)
    p.add_argument("--min-gap-atr", type=float, default=0.08)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    params = Params(
        rr=args.rr,
        min_gap_atr=args.min_gap_atr,
        require_ob_overlap=args.require_ob_overlap,
        require_liquidity_overlap=args.require_liquidity_overlap,
        require_h1_trend=args.require_h1_trend,
        start=args.start,
        end=args.end,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    df = add_h1_trend(add_atr(load_data(args.data, args.start, args.end), params.atr_period))
    summary, trades, counters = backtest(df, params)
    summary_path = args.out / "zone_retest_summary.json"
    trades_path = args.out / "zone_retest_trades.csv"
    diagnostics_path = args.out / "zone_retest_diagnostics.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    trades.to_csv(trades_path, index=False)
    diagnostics = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source_csv": str(args.data),
        "params": dataclasses.asdict(params),
        "counters": counters,
    }
    diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_design(args.out, args.data, summary, counters)
    print(json.dumps({"summary": summary, "counters": counters, "out_dir": str(args.out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
