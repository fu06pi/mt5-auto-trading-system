#!/usr/bin/env python3
"""Fast research backtest for the MT5 IFVG Sniper Python port.

Research-only. Uses cached XAUUSD M5 OHLC data and does not touch live MT5 state.
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

DATA_MT5 = Path("/home/chain4655/Documents/backtest_reports/xauusd_5y_weekly_reset/data/XAUUSD_M5_5y_mt5.csv")
OUT = Path("/home/chain4655/Documents/backtest_reports/ifvg_sniper_backtest")

INITIAL_EQUITY = 10000.0
POINT = 0.01
CONTRACT = 100.0
FIXED_LOTS = 0.01
SPREAD_PRICE = 0.39
COMMISSION_PER_LOT_ROUND = 7.0


@dataclasses.dataclass(frozen=True)
class Params:
    name: str
    filter_mode: str = "balanced"
    rr: float = 3.0
    sl_atr_mult: float = 1.5
    entry_basis: str = "ifvg_line"
    max_hidden_fvg: int = 120
    max_fvg_age: int = 60
    min_gap_ticks: int = 0
    atr_period: int = 14
    max_trades_day: int = 5
    max_hold_bars: int = 288


@dataclasses.dataclass(frozen=True)
class Signal:
    i: int
    direction: int
    top: float
    bot: float
    close: float
    line_price: float
    gap_atr: float
    body_ratio: float
    range_atr: float


@dataclasses.dataclass
class Position:
    direction: int
    entry_time: pd.Timestamp
    entry_i: int
    entry: float
    sl: float
    tp: float
    lots: float


class FastIFVGEngine:
    def __init__(self, p: Params, highs: List[float], lows: List[float], opens: List[float], closes: List[float], atr: List[float]) -> None:
        self.p = p
        self.highs = highs
        self.lows = lows
        self.opens = opens
        self.closes = closes
        self.atr = atr
        self.raw_tops: List[float] = []
        self.raw_bots: List[float] = []
        self.raw_dirs: List[int] = []
        self.raw_ages: List[int] = []
        self.raw_gap_atr: List[float] = []
        self.raw_body_ratio: List[float] = []
        self.raw_range_atr: List[float] = []
        self.filtered = 0

    def step(self, i: int) -> Optional[Signal]:
        if i < 2:
            return None
        safe_atr = self.safe_atr(i)
        for pos in range(len(self.raw_ages)):
            self.raw_ages[pos] += 1
        for pos in range(len(self.raw_ages) - 1, -1, -1):
            if self.raw_ages[pos] > self.p.max_fvg_age:
                self.remove(pos)

        candle_range = max(self.highs[i] - self.lows[i], POINT)
        candle_body = abs(self.closes[i] - self.opens[i])
        body_ratio = candle_body / candle_range
        range_atr_ratio = candle_range / safe_atr
        min_gap = self.p.min_gap_ticks * POINT

        raw_bull = self.lows[i] > self.highs[i - 2] and (self.lows[i] - self.highs[i - 2]) >= min_gap
        raw_bear = self.highs[i] < self.lows[i - 2] and (self.lows[i - 2] - self.highs[i]) >= min_gap
        if raw_bull:
            self.raw_tops.append(self.lows[i])
            self.raw_bots.append(self.highs[i - 2])
            self.raw_dirs.append(1)
            self.raw_ages.append(0)
            self.raw_gap_atr.append((self.lows[i] - self.highs[i - 2]) / safe_atr)
            self.raw_body_ratio.append(body_ratio)
            self.raw_range_atr.append(range_atr_ratio)
        if raw_bear:
            self.raw_tops.append(self.lows[i - 2])
            self.raw_bots.append(self.highs[i])
            self.raw_dirs.append(-1)
            self.raw_ages.append(0)
            self.raw_gap_atr.append((self.lows[i - 2] - self.highs[i]) / safe_atr)
            self.raw_body_ratio.append(body_ratio)
            self.raw_range_atr.append(range_atr_ratio)
        while len(self.raw_tops) > self.p.max_hidden_fvg:
            self.remove(0)

        for pos in range(len(self.raw_tops) - 1, -1, -1):
            top = self.raw_tops[pos]
            bot = self.raw_bots[pos]
            raw_dir = self.raw_dirs[pos]
            buf = safe_atr * self.break_buffer_atr()
            bull_inv = raw_dir == -1 and self.closes[i] > top + buf
            bear_inv = raw_dir == 1 and self.closes[i] < bot - buf
            if bull_inv or bear_inv:
                signal = None
                if self.quality_pass(self.raw_gap_atr[pos], self.raw_body_ratio[pos], self.raw_range_atr[pos]):
                    direction = 1 if bull_inv else -1
                    line_price = top if direction == 1 else bot
                    signal = Signal(i, direction, top, bot, self.closes[i], line_price, self.raw_gap_atr[pos], self.raw_body_ratio[pos], self.raw_range_atr[pos])
                else:
                    self.filtered += 1
                self.remove(pos)
                return signal
        return None

    def remove(self, pos: int) -> None:
        del self.raw_tops[pos]
        del self.raw_bots[pos]
        del self.raw_dirs[pos]
        del self.raw_ages[pos]
        del self.raw_gap_atr[pos]
        del self.raw_body_ratio[pos]
        del self.raw_range_atr[pos]

    def safe_atr(self, i: int) -> float:
        val = self.atr[i]
        return POINT if math.isnan(val) or val <= 0 else val

    def min_gap_atr(self) -> float:
        return {"off": 0.0, "loose": 0.15, "balanced": 0.25, "strict": 0.40}.get(self.p.filter_mode, 0.25)

    def min_body_ratio(self) -> float:
        return {"off": 0.0, "loose": 0.40, "balanced": 0.50, "strict": 0.60}.get(self.p.filter_mode, 0.50)

    def min_range_atr(self) -> float:
        return {"off": 0.0, "loose": 0.40, "balanced": 0.60, "strict": 0.85}.get(self.p.filter_mode, 0.60)

    def break_buffer_atr(self) -> float:
        return {"off": 0.0, "loose": 0.0, "balanced": 0.05, "strict": 0.10}.get(self.p.filter_mode, 0.05)

    def quality_pass(self, gap_atr: float, body_ratio: float, range_atr: float) -> bool:
        return self.p.filter_mode == "off" or (gap_atr >= self.min_gap_atr() and body_ratio >= self.min_body_ratio() and range_atr >= self.min_range_atr())


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)[["time", "open", "high", "low", "close"]]


def atr_values(df: pd.DataFrame, period: int) -> List[float]:
    high, low, close = df.high.tolist(), df.low.tolist(), df.close.tolist()
    trs = []
    for i in range(len(df)):
        trs.append(high[i] - low[i] if i == 0 else max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])))
    out = [math.nan] * len(df)
    if len(df) >= period:
        prev = sum(trs[:period]) / period
        out[period - 1] = prev
        for i in range(period, len(df)):
            prev = (prev * (period - 1) + trs[i]) / period
            out[i] = prev
    return out


def pnl_usd(direction: int, entry: float, exit_price: float, lots: float) -> float:
    gross = (exit_price - entry) * direction * CONTRACT * lots
    cost = SPREAD_PRICE * CONTRACT * lots + COMMISSION_PER_LOT_ROUND * lots
    return gross - cost


def backtest(df: pd.DataFrame, params: Params) -> Tuple[Dict[str, Any], pd.DataFrame]:
    highs, lows, opens, closes = df.high.tolist(), df.low.tolist(), df.open.tolist(), df.close.tolist()
    atr = atr_values(df, params.atr_period)
    engine = FastIFVGEngine(params, highs, lows, opens, closes, atr)
    equity = INITIAL_EQUITY
    peak = equity
    max_dd = 0.0
    pos: Optional[Position] = None
    trades: List[Dict[str, Any]] = []
    trades_day: Dict[str, int] = {}
    signals = blocked = 0

    for i in range(params.atr_period + 5, len(df)):
        bar = df.iloc[i]
        if pos is not None:
            hit_sl = lows[i] <= pos.sl if pos.direction == 1 else highs[i] >= pos.sl
            hit_tp = highs[i] >= pos.tp if pos.direction == 1 else lows[i] <= pos.tp
            reason = ""
            exit_price = 0.0
            if hit_sl:  # conservative: SL before TP when both hit same candle
                reason, exit_price = "SL", pos.sl
            elif hit_tp:
                reason, exit_price = "TP", pos.tp
            elif i - pos.entry_i >= params.max_hold_bars:
                reason, exit_price = "TIME", closes[i]
            if reason:
                profit = pnl_usd(pos.direction, pos.entry, exit_price, pos.lots)
                equity += profit
                peak = max(peak, equity)
                max_dd = max(max_dd, (peak - equity) / peak if peak else 0.0)
                trades.append({"entry_time": pos.entry_time, "exit_time": bar.time, "side": "BUY" if pos.direction == 1 else "SELL", "entry": pos.entry, "sl": pos.sl, "tp": pos.tp, "exit": exit_price, "reason": reason, "lots": pos.lots, "pnl": profit, "equity": equity, "bars_held": i - pos.entry_i})
                pos = None

        sig = engine.step(i)
        if sig is None:
            continue
        signals += 1
        day = str(bar.time.date())
        if pos is not None or trades_day.get(day, 0) >= params.max_trades_day:
            blocked += 1
            continue
        safe_atr = engine.safe_atr(i)
        entry = closes[i] if params.entry_basis == "close" else sig.line_price
        risk = safe_atr * params.sl_atr_mult
        sl = entry - risk if sig.direction == 1 else entry + risk
        tp = entry + risk * params.rr if sig.direction == 1 else entry - risk * params.rr
        pos = Position(sig.direction, bar.time, i, entry, sl, tp, FIXED_LOTS)
        trades_day[day] = trades_day.get(day, 0) + 1

    if pos is not None:
        bar = df.iloc[-1]
        profit = pnl_usd(pos.direction, pos.entry, closes[-1], pos.lots)
        equity += profit
        trades.append({"entry_time": pos.entry_time, "exit_time": bar.time, "side": "BUY" if pos.direction == 1 else "SELL", "entry": pos.entry, "sl": pos.sl, "tp": pos.tp, "exit": closes[-1], "reason": "EOD", "lots": pos.lots, "pnl": profit, "equity": equity, "bars_held": len(df) - 1 - pos.entry_i})

    tdf = pd.DataFrame(trades)
    wins = int((tdf.pnl > 0).sum()) if not tdf.empty else 0
    losses = int((tdf.pnl <= 0).sum()) if not tdf.empty else 0
    gp = float(tdf.loc[tdf.pnl > 0, "pnl"].sum()) if not tdf.empty else 0.0
    gl = float(-tdf.loc[tdf.pnl <= 0, "pnl"].sum()) if not tdf.empty else 0.0
    summary = {
        "name": params.name,
        "filter_mode": params.filter_mode,
        "rr": params.rr,
        "entry_basis": params.entry_basis,
        "bars": len(df),
        "from": str(df.time.iloc[0]),
        "to": str(df.time.iloc[-1]),
        "signals": signals,
        "blocked": blocked,
        "filtered_ifvg": engine.filtered,
        "trades": len(tdf),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(tdf) if len(tdf) else 0.0,
        "net_pnl": float(tdf.pnl.sum()) if not tdf.empty else 0.0,
        "final_equity": equity,
        "profit_factor": gp / gl if gl > 0 else None,
        "max_dd_pct": max_dd,
        "avg_pnl": float(tdf.pnl.mean()) if not tdf.empty else 0.0,
    }
    return summary, tdf


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_data(DATA_MT5)
    params_list = [Params(f"{entry}_{mode}_{rr:g}r", mode, rr, entry_basis=entry) for entry in ["ifvg_line", "close"] for mode in ["off", "loose", "balanced", "strict"] for rr in [1.0, 2.0, 3.0]]
    summaries = []
    for params in params_list:
        summary, trades = backtest(df, params)
        summaries.append(summary)
        trades.to_csv(OUT / f"{params.name}_trades.csv", index=False)
    sdf = pd.DataFrame(summaries).sort_values("net_pnl", ascending=False)
    sdf.to_csv(OUT / "summary.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(sdf.head(12).to_string(index=False))
    print(f"OUT={OUT}")


if __name__ == "__main__":
    main()
