#!/usr/bin/env python3.14
"""Backtest for a Tide-Wave Grid strategy on XAUUSD.

Research note:
No exact public rule-set for the Chinese phrase "潮汐波浪网格量化策略" was found via
Bing/DuckDuckGo at implementation time, so this script implements a testable composite:
- Tide: higher-level EMA trend/slope regime classifier.
- Wave: RSI + z-score distance from a moving fair-value center.
- Grid: ATR-spaced mean-reversion ladder with basket exits and hard risk stops.

This is a research prototype, not a live MT5 strategy.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from pymt5linux import MetaTrader5  # type: ignore[import-not-found]
except ImportError:
    sys.path.insert(0, "/home/chain4655/.openharness-venv/lib64/python3.14/site-packages")
    from pymt5linux import MetaTrader5  # type: ignore[import-not-found]

ROOT = Path("/home/chain4655/Documents/Projects/MT5")
OUT_DIR = ROOT / "backtest_reports_tide_wave_grid"
SYMBOL = "XAUUSD"
TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
INITIAL_EQUITY = 10000.0
CONTRACT_SIZE = 100.0
POINT = 0.01
SPREAD_POINTS = 20.0
COMMISSION_PER_LOT = 7.0


@dataclasses.dataclass(frozen=True)
class Bar:
    time: dt.datetime
    open: float
    high: float
    low: float
    close: float
    tick_volume: float = 0.0


@dataclasses.dataclass
class Leg:
    side: str
    entry: float
    volume: float
    level: int
    entry_time: dt.datetime


@dataclasses.dataclass
class BasketTrade:
    strategy: str
    timeframe: str
    entry_time: dt.datetime
    exit_time: dt.datetime
    side: str
    legs: int
    avg_entry: float
    exit: float
    total_volume: float
    pnl_gross: float
    pnl_net: float
    commission: float
    spread_cost: float
    reason: str
    regime: str
    atr: float
    bars_held: int
    max_adverse_atr: float
    max_favorable_atr: float


@dataclasses.dataclass(frozen=True)
class TideWaveGridConfig:
    name: str
    timeframe: str = "M15"
    ema_center: int = 34
    ema_tide_fast: int = 50
    ema_tide_slow: int = 144
    atr_period: int = 14
    rsi_period: int = 14
    z_window: int = 80
    grid_step_atr: float = 0.80
    take_profit_atr: float = 0.35
    basket_center_exit_atr: float = 0.10
    max_levels: int = 4
    base_lot: float = 0.03
    lot_multiplier: float = 1.35
    max_total_lots: float = 0.35
    trend_slope_atr: float = 0.18
    range_slope_atr: float = 0.08
    z_entry: float = 0.75
    rsi_buy: float = 42.0
    rsi_sell: float = 58.0
    hard_stop_atr: float = 3.8
    max_hold_bars: int = 96
    cooldown_bars: int = 4
    allow_trend_counter: bool = False


def fetch_mt5_bars(timeframe: str, days: int = 180) -> Tuple[List[Bar], str]:
    mt5 = MetaTrader5(host="127.0.0.1", port=18812)
    ok = mt5.initialize(path=TERMINAL_PATH)
    if not ok:
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
        }
        mtf = tf_map[timeframe]
        end = dt.datetime.now()
        start = end - dt.timedelta(days=days)
        rates = mt5.copy_rates_range(SYMBOL, mtf, start, end)
        if rates is None or len(rates) == 0:
            fallback = {"M1": 180000, "M5": 52000, "M15": 18000, "H1": 5000}
            rates = mt5.copy_rates_from_pos(SYMBOL, mtf, 0, fallback[timeframe])
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No rates returned: {mt5.last_error()}")
        bars = [
            Bar(
                time=dt.datetime.fromtimestamp(int(row["time"])),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                tick_volume=float(row["tick_volume"]),
            )
            for row in rates
        ]
        bars.sort(key=lambda b: b.time)
        return bars, infer_timeframe(bars) or timeframe
    finally:
        try:
            mt5.shutdown()
        except RuntimeError:
            pass


def infer_timeframe(bars: Sequence[Bar]) -> Optional[str]:
    if len(bars) < 2:
        return None
    delta = int((bars[1].time - bars[0].time).total_seconds())
    return {60: "M1", 300: "M5", 900: "M15", 3600: "H1"}.get(delta)


def ema_series(values: Sequence[float], period: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(values)
    if len(values) < period:
        return out
    alpha = 2.0 / (period + 1.0)
    ema = statistics.fmean(values[:period])
    out[period - 1] = ema
    for i in range(period, len(values)):
        ema = values[i] * alpha + ema * (1.0 - alpha)
        out[i] = ema
    return out


def atr_series(bars: Sequence[Bar], period: int) -> List[Optional[float]]:
    trs: List[float] = []
    out: List[Optional[float]] = [None] * len(bars)
    for i, bar in enumerate(bars):
        if i == 0:
            tr = bar.high - bar.low
        else:
            prev = bars[i - 1]
            tr = max(bar.high - bar.low, abs(bar.high - prev.close), abs(bar.low - prev.close))
        trs.append(tr)
        if i >= period:
            out[i] = max(statistics.fmean(trs[i - period + 1:i + 1]), POINT * 5)
    return out


def rsi_series(values: Sequence[float], period: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(values)
    if len(values) <= period:
        return out
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, period + 1):
        diff = values[i] - values[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    avg_gain = statistics.fmean(gains)
    avg_loss = statistics.fmean(losses)
    out[period] = 100.0 if avg_loss == 0 else 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
    for i in range(period + 1, len(values)):
        diff = values[i] - values[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(diff, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-diff, 0.0)) / period
        out[i] = 100.0 if avg_loss == 0 else 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
    return out


def rolling_z(values: Sequence[float], center: Sequence[Optional[float]], window: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(values)
    distances: List[float] = []
    for i, value in enumerate(values):
        center_i = center[i]
        if center_i is None:
            distances.append(0.0)
            continue
        distance = value - float(center_i)
        distances.append(distance)
        if i >= window:
            sample = distances[i - window + 1:i + 1]
            sd = statistics.pstdev(sample)
            if sd > 1e-9:
                out[i] = distance / sd
    return out


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def compute_leg_pnl(leg: Leg, exit_price: float) -> float:
    mult = 1.0 if leg.side == "BUY" else -1.0
    return (exit_price - leg.entry) * mult * leg.volume * CONTRACT_SIZE


def weighted_avg(legs: Sequence[Leg]) -> float:
    volume = sum(leg.volume for leg in legs)
    if volume <= 0:
        return 0.0
    return sum(leg.entry * leg.volume for leg in legs) / volume


def lot_for_level(cfg: TideWaveGridConfig, level: int, current_total: float) -> float:
    raw = cfg.base_lot * (cfg.lot_multiplier ** max(level - 1, 0))
    lot = math.floor(raw / 0.01) * 0.01
    remaining = max(cfg.max_total_lots - current_total, 0.0)
    return round(max(0.0, min(lot, remaining)), 2)


def classify_regime(
    close: float,
    ema_fast: float,
    ema_slow: float,
    atr_value: float,
    cfg: TideWaveGridConfig,
) -> str:
    slope = (ema_fast - ema_slow) / max(atr_value, 1e-9)
    if slope >= cfg.trend_slope_atr and close >= ema_fast:
        return "bull_tide"
    if slope <= -cfg.trend_slope_atr and close <= ema_fast:
        return "bear_tide"
    if abs(slope) <= cfg.range_slope_atr:
        return "slack_tide_range"
    return "mixed_tide"


def allowed_side(regime: str, cfg: TideWaveGridConfig, side: str) -> bool:
    if cfg.allow_trend_counter:
        return True
    if regime == "bull_tide":
        return side == "BUY"
    if regime == "bear_tide":
        return side == "SELL"
    return True


def backtest_grid(bars: Sequence[Bar], cfg: TideWaveGridConfig) -> Tuple[List[BasketTrade], Dict[str, Any]]:
    closes = [bar.close for bar in bars]
    center = ema_series(closes, cfg.ema_center)
    tide_fast = ema_series(closes, cfg.ema_tide_fast)
    tide_slow = ema_series(closes, cfg.ema_tide_slow)
    atr_vals = atr_series(bars, cfg.atr_period)
    rsi_vals = rsi_series(closes, cfg.rsi_period)
    z_vals = rolling_z(closes, center, cfg.z_window)

    equity = INITIAL_EQUITY
    peak = equity
    max_dd = 0.0
    trades: List[BasketTrade] = []
    legs: List[Leg] = []
    current_side: Optional[str] = None
    entry_index: Optional[int] = None
    entry_regime = ""
    entry_atr = 0.0
    cooldown_until = -1
    max_adverse = 0.0
    max_favorable = 0.0

    warmup = max(cfg.ema_tide_slow + 5, cfg.z_window + 5, cfg.atr_period + 5)
    for i in range(warmup, len(bars) - 1):
        bar = bars[i]
        nxt = bars[i + 1]
        center_i = center[i]
        tide_fast_i = tide_fast[i]
        tide_slow_i = tide_slow[i]
        atr_i = atr_vals[i]
        rsi_i = rsi_vals[i]
        z_i = z_vals[i]
        if center_i is None or tide_fast_i is None or tide_slow_i is None:
            continue
        if atr_i is None or rsi_i is None or z_i is None:
            continue
        atr_value = float(atr_i)
        center_value = float(center_i)
        regime = classify_regime(bar.close, float(tide_fast_i), float(tide_slow_i), atr_value, cfg)
        step = max(atr_value * cfg.grid_step_atr, POINT * 20)

        if legs:
            assert current_side is not None and entry_index is not None
            avg_entry = weighted_avg(legs)
            total_volume = sum(leg.volume for leg in legs)
            mult = 1.0 if current_side == "BUY" else -1.0
            favorable_atr = (bar.close - avg_entry) * mult / max(atr_value, 1e-9)
            adverse_atr = -favorable_atr
            max_favorable = max(max_favorable, favorable_atr)
            max_adverse = max(max_adverse, adverse_atr)

            exit_price: Optional[float] = None
            reason = ""
            target = avg_entry + mult * max(cfg.take_profit_atr * atr_value, POINT * 30)
            center_exit_ok = (bar.close - center_value) * mult >= -cfg.basket_center_exit_atr * atr_value
            if (current_side == "BUY" and nxt.high >= target) or (current_side == "SELL" and nxt.low <= target):
                exit_price = target
                reason = "BASKET_TP"
            elif center_exit_ok and (len(legs) >= 2 or favorable_atr >= 0):
                exit_price = nxt.open
                reason = "CENTER_REVERT"
            elif adverse_atr >= cfg.hard_stop_atr:
                exit_price = nxt.open
                reason = "HARD_STOP"
            elif i - entry_index >= cfg.max_hold_bars:
                exit_price = nxt.open
                reason = "TIME_STOP"

            if exit_price is not None:
                pnl_gross = sum(compute_leg_pnl(leg, exit_price) for leg in legs)
                spread_cost = SPREAD_POINTS * POINT * total_volume * CONTRACT_SIZE
                commission = total_volume * COMMISSION_PER_LOT
                pnl_net = pnl_gross - spread_cost - commission
                equity += pnl_net
                peak = max(peak, equity)
                max_dd = min(max_dd, equity / peak - 1.0)
                trades.append(
                    BasketTrade(
                        strategy=cfg.name,
                        timeframe=cfg.timeframe,
                        entry_time=legs[0].entry_time,
                        exit_time=nxt.time,
                        side=current_side,
                        legs=len(legs),
                        avg_entry=avg_entry,
                        exit=exit_price,
                        total_volume=total_volume,
                        pnl_gross=pnl_gross,
                        pnl_net=pnl_net,
                        commission=commission,
                        spread_cost=spread_cost,
                        reason=reason,
                        regime=entry_regime,
                        atr=entry_atr,
                        bars_held=i - entry_index,
                        max_adverse_atr=max_adverse,
                        max_favorable_atr=max_favorable,
                    )
                )
                legs = []
                current_side = None
                entry_index = None
                entry_regime = ""
                entry_atr = 0.0
                max_adverse = 0.0
                max_favorable = 0.0
                cooldown_until = i + cfg.cooldown_bars
                continue

            if len(legs) < cfg.max_levels:
                next_level = len(legs) + 1
                distance = (avg_entry - bar.close) if current_side == "BUY" else (bar.close - avg_entry)
                if distance >= step * next_level:
                    current_total = sum(leg.volume for leg in legs)
                    lot = lot_for_level(cfg, next_level, current_total)
                    if lot >= 0.01:
                        legs.append(Leg(current_side, nxt.open, lot, next_level, nxt.time))
            continue

        if i < cooldown_until:
            continue
        z_value = float(z_i)
        rsi_value = float(rsi_i)
        side: Optional[str] = None
        if z_value <= -cfg.z_entry and rsi_value <= cfg.rsi_buy:
            side = "BUY"
        elif z_value >= cfg.z_entry and rsi_value >= cfg.rsi_sell:
            side = "SELL"
        if side is None or not allowed_side(regime, cfg, side):
            continue
        lot = lot_for_level(cfg, 1, 0.0)
        if lot >= 0.01:
            legs = [Leg(side, nxt.open, lot, 1, nxt.time)]
            current_side = side
            entry_index = i + 1
            entry_regime = regime
            entry_atr = atr_value

    wins = [trade for trade in trades if trade.pnl_net > 0]
    losses = [trade for trade in trades if trade.pnl_net <= 0]
    gross_profit = sum(trade.pnl_net for trade in wins)
    gross_loss = -sum(trade.pnl_net for trade in losses)
    metrics: Dict[str, Any] = {
        "strategy": cfg.name,
        "timeframe": cfg.timeframe,
        "bars": len(bars),
        "start": bars[0].time.isoformat() if bars else "",
        "end": bars[-1].time.isoformat() if bars else "",
        "trades": len(trades),
        "net_pnl": round(equity - INITIAL_EQUITY, 2),
        "return_pct": round((equity / INITIAL_EQUITY - 1.0) * 100.0, 2),
        "win_rate_pct": round(len(wins) / len(trades) * 100.0, 2) if trades else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0),
        "max_dd_pct": round(max_dd * 100.0, 2),
        "expectancy_usd": round(statistics.fmean([trade.pnl_net for trade in trades]), 2) if trades else 0.0,
        "avg_legs": round(statistics.fmean([trade.legs for trade in trades]), 2) if trades else 0.0,
        "avg_hold_bars": round(statistics.fmean([trade.bars_held for trade in trades]), 2) if trades else 0.0,
        "total_costs": round(sum(trade.commission + trade.spread_cost for trade in trades), 2),
        "open_basket_ignored": bool(legs),
    }
    return trades, metrics


def write_trades(path: Path, trades: Sequence[BasketTrade]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in dataclasses.fields(BasketTrade)])
        writer.writeheader()
        for trade in trades:
            row = dataclasses.asdict(trade)
            row["entry_time"] = trade.entry_time.isoformat()
            row["exit_time"] = trade.exit_time.isoformat()
            writer.writerow(row)


def print_row(metrics: Dict[str, Any]) -> None:
    print(
        f"{metrics['strategy']:<26s} {metrics['timeframe']:>4s} "
        f"trades={metrics['trades']:>4d} pnl={metrics['net_pnl']:>9.2f} "
        f"ret={metrics['return_pct']:>7.2f}% wr={metrics['win_rate_pct']:>5.1f}% "
        f"pf={metrics['profit_factor']:>6.3f} dd={metrics['max_dd_pct']:>7.2f}% "
        f"exp={metrics['expectancy_usd']:>7.2f} legs={metrics['avg_legs']:>4.2f}"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configs = [
        TideWaveGridConfig(name="tide_wave_grid_m15_base", timeframe="M15"),
        TideWaveGridConfig(
            name="tide_wave_grid_m15_research_best",
            timeframe="M15",
            grid_step_atr=0.80,
            z_entry=0.80,
            rsi_buy=38.0,
            rsi_sell=62.0,
            take_profit_atr=1.20,
            hard_stop_atr=3.0,
            max_levels=3,
            base_lot=0.02,
            max_total_lots=0.12,
            allow_trend_counter=False,
        ),
        TideWaveGridConfig(
            name="tide_wave_grid_m15_selective",
            timeframe="M15",
            grid_step_atr=0.80,
            z_entry=2.00,
            rsi_buy=38.0,
            rsi_sell=62.0,
            take_profit_atr=0.90,
            hard_stop_atr=3.0,
            max_levels=3,
            base_lot=0.02,
            max_total_lots=0.12,
            allow_trend_counter=False,
        ),
        TideWaveGridConfig(
            name="tide_wave_grid_m5_base",
            timeframe="M5",
            ema_center=55,
            ema_tide_fast=89,
            ema_tide_slow=233,
            z_window=120,
            grid_step_atr=0.90,
            max_hold_bars=180,
        ),
        TideWaveGridConfig(
            name="tide_wave_grid_m5_defensive",
            timeframe="M5",
            ema_center=55,
            ema_tide_fast=89,
            ema_tide_slow=233,
            z_window=120,
            grid_step_atr=1.05,
            max_levels=3,
            base_lot=0.02,
            max_total_lots=0.18,
            hard_stop_atr=3.0,
            max_hold_bars=144,
        ),
    ]
    bars_by_tf: Dict[str, List[Bar]] = {}
    for timeframe in sorted({cfg.timeframe for cfg in configs}):
        bars, actual = fetch_mt5_bars(timeframe=timeframe, days=180)
        bars_by_tf[timeframe] = bars
        print(f"Fetched {len(bars)} {SYMBOL} bars for requested {timeframe}; actual={actual}; {bars[0].time} -> {bars[-1].time}")

    summary: Dict[str, Any] = {
        "symbol": SYMBOL,
        "initial_equity": INITIAL_EQUITY,
        "spread_points": SPREAD_POINTS,
        "commission_per_lot": COMMISSION_PER_LOT,
        "research_note": "No exact public rule-set found; implemented Tide=EMA trend, Wave=RSI/zscore, Grid=ATR ladder.",
        "strategies": {},
    }
    for cfg in configs:
        trades, metrics = backtest_grid(bars_by_tf[cfg.timeframe], cfg)
        write_trades(OUT_DIR / f"{cfg.name}_trades.csv", trades)
        summary["strategies"][cfg.name] = {"config": dataclasses.asdict(cfg), "metrics": metrics}
        print_row(metrics)

    with (OUT_DIR / "tide_wave_grid_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    with (OUT_DIR / "tide_wave_grid_summary.csv").open("w", newline="", encoding="utf-8") as f:
        rows = [data["metrics"] for data in summary["strategies"].values()]
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary saved: {OUT_DIR / 'tide_wave_grid_summary.json'}")
    print(f"CSV saved: {OUT_DIR / 'tide_wave_grid_summary.csv'}")


if __name__ == "__main__":
    main()
