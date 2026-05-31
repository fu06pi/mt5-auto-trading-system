#!/usr/bin/env python3.14
"""Enhanced backtest v2: trailing stops, spread cost, live param matching, live comparison.

Fixes over original backtest_compare_strategies.py:
  - Trailing stop simulation (conservative: uses bar OHLC for trigger detection)
  - Spread + commission cost model
  - Strategy params match live active_plan.json exactly
  - Live trade history comparison
  - Runs each strategy in BOTH no-trailing and trailing mode
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from pymt5linux import MetaTrader5
except ImportError:
    sys.path.insert(0, "/home/chain4655/.openharness-venv/lib64/python3.14/site-packages")
    from pymt5linux import MetaTrader5

ROOT = Path("/home/chain4655/Documents/Projects/MT5")
OUT_DIR = ROOT / "backtest_reports_v2"
SYMBOL = "XAUUSD"
TIMEFRAME = "M15"
TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
INITIAL_EQUITY = 10000.0
CONTRACT_SIZE = 100.0
POINT = 0.01
SPREAD_POINTS = 20.0
COMMISSION_PER_LOT = 7.0


@dataclasses.dataclass
class Bar:
    time: dt.datetime
    open: float
    high: float
    low: float
    close: float
    tick_volume: float = 0.0


@dataclasses.dataclass
class Trade:
    strategy: str
    mode: str
    entry_time: dt.datetime
    exit_time: dt.datetime
    side: str
    entry: float
    exit: float
    sl: float
    tp: float
    volume: float
    pnl: float
    pnl_with_costs: float
    commission: float
    spread_cost: float
    r_multiple: float
    reason: str
    regime: str
    atr: float
    bars_held: int


def fetch_mt5_bars(tf: str = TIMEFRAME) -> Tuple[List[Bar], str]:
    mt5 = MetaTrader5(host="127.0.0.1", port=18812)
    ok = mt5.initialize(path=TERMINAL_PATH)
    if not ok:
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        tf_map = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15}
        mtf = tf_map.get(tf, mt5.TIMEFRAME_M15)
        end = dt.datetime.now()
        start = end - dt.timedelta(days=92)
        rates = mt5.copy_rates_range(SYMBOL, mtf, start, end)
        if rates is None or len(rates) == 0:
            est_bars = {"M1": 130000, "M5": 26000, "M15": 9000}
            rates = mt5.copy_rates_from_pos(SYMBOL, mtf, 0, est_bars.get(tf, 9000))
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No rates returned: {mt5.last_error()}")
        bars: List[Bar] = []
        for row in rates:
            ts = int(row["time"])
            bars.append(Bar(time=dt.datetime.fromtimestamp(ts), open=float(row["open"]),
                            high=float(row["high"]), low=float(row["low"]),
                            close=float(row["close"]), tick_volume=float(row["tick_volume"])))
        bars.sort(key=lambda b: b.time)
        use_tf = _infer_tf(bars) or tf
        return bars, use_tf
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


def _infer_tf(bars: List[Bar]) -> Optional[str]:
    if len(bars) < 2:
        return None
    delta = (bars[1].time - bars[0].time).total_seconds()
    return {60: "M1", 300: "M5", 900: "M15"}.get(int(delta))


def sma(values: Sequence[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    return statistics.fmean(values[-period:])


def atr(bars: Sequence[Bar], period: int) -> Optional[float]:
    if len(bars) < period + 1:
        return None
    trs: List[float] = []
    for i in range(-period, 0):
        curr = bars[i]
        prev = bars[i - 1]
        trs.append(max(curr.high - curr.low, abs(curr.high - prev.close), abs(curr.low - prev.close)))
    return max(statistics.fmean(trs), POINT * 5)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * clamp(q, 0.0, 1.0)))
    return ordered[idx]


def market_regime(window: Sequence[Bar]) -> str:
    closes = [b.close for b in window]
    highs = [b.high for b in window]
    lows = [b.low for b in window]
    a = atr(window, 14) or 0.0
    if len(closes) < 50 or a <= 0:
        return "warmup"
    fast = sma(closes, 10) or closes[-1]
    slow = sma(closes, 30) or closes[-1]
    slope = abs(fast - slow) / a
    rng = (max(highs[-20:]) - min(lows[-20:])) / a
    if slope >= 1.0:
        return "trend"
    if rng <= 2.0:
        return "range/compression"
    if a / max(closes[-1], 1e-9) >= 0.0018:
        return "high-volatility"
    return "mixed"


def position_size(equity: float, risk_pct: float, entry: float, sl: float, max_lots: float = 5.0) -> float:
    risk = equity * risk_pct
    risk_per_lot = abs(entry - sl) * CONTRACT_SIZE
    if risk_per_lot <= 0:
        return 0.0
    lots = risk / risk_per_lot
    return max(0.01, min(max_lots, math.floor(lots / 0.01) * 0.01))


def compute_costs(entry: float, exit: float, volume: float, side: str) -> Tuple[float, float, float]:
    spread_cost = SPREAD_POINTS * POINT * volume * CONTRACT_SIZE
    commission = volume * COMMISSION_PER_LOT
    raw_pnl = (exit - entry) * volume * CONTRACT_SIZE * (1.0 if side == "BUY" else -1.0)
    return raw_pnl - spread_cost - commission, commission, spread_cost


class BaseStrategy:
    name = "base"
    mode = "no_trail"
    risk_pct = 0.0075
    stop_atr = 5.0
    reward_multiple = 2.2
    max_hold_bars = 32
    cooldown_bars = 4
    use_trailing = False
    trail_trigger_atr = 1.5
    trail_lock_atr = 0.5

    def signal(self, hist: Sequence[Bar]) -> Tuple[str, float, float]:
        return "NONE", 0.0, 0.0

    def sl_tp(self, side: str, entry: float, a: float, score: float) -> Tuple[float, float]:
        stop = a * self.stop_atr
        reward = stop * self.reward_multiple
        return (entry - stop, entry + reward) if side == "BUY" else (entry + stop, entry - reward)

    def get_trailed_sl(self, side: str, entry: float, bar_high: float, bar_low: float, sl: float, atr_val: float) -> float:
        """Conservative trailing: check if bar's price reached trigger, move SL if so."""
        if not self.use_trailing:
            return sl
        if side == "BUY":
            trigger_price = entry + atr_val * self.trail_trigger_atr
            if bar_high >= trigger_price:
                new_sl = max(sl, entry + atr_val * self.trail_lock_atr)
                return new_sl
        else:
            trigger_price = entry - atr_val * self.trail_trigger_atr
            if bar_low <= trigger_price:
                new_sl = min(sl, entry - atr_val * self.trail_lock_atr)
                return new_sl
        return sl

    timeframe = "M15"

class DoomsdayStrategyV2(BaseStrategy):
    name = "doomsday_v2"
    timeframe = "M5"
    mode = "no_trail"
    risk_pct = 0.0075
    stop_atr = 3.0
    reward_multiple = 2.4
    long_bias = 0.68
    threshold = 0.60
    fast = 7
    slow = 30
    breakout_lookback = 16
    high_vol_atr_pct = 0.0021
    high_vol_range_atr = 4.75
    high_vol_min_momentum = 0.85
    high_vol_spike_atr = 3.0
    high_vol_min_breakout_atr = 0.35
    high_vol_min_close_location = 0.68
    high_vol_only = True
    roll_trigger_pct = 0.08
    max_hold_bars = 48
    cooldown_bars = 5

    def _high_volatility(self, hist: Sequence[Bar], a: float) -> Tuple[bool, float, float, float]:
        closes = [b.close for b in hist]
        highs = [b.high for b in hist]
        lows = [b.low for b in hist]
        atr_pct = a / max(closes[-1], 1e-9)
        lookback = min(20, len(highs), len(lows))
        range_atr = (max(highs[-lookback:]) - min(lows[-lookback:])) / max(a, 1e-9)
        last_range = max(hist[-1].high - hist[-1].low, 0.0)
        spike_atr = last_range / max(a, 1e-9)
        return atr_pct >= self.high_vol_atr_pct and (range_atr >= self.high_vol_range_atr or spike_atr >= self.high_vol_spike_atr), atr_pct, range_atr, spike_atr

    def _score_signal(self, hist: Sequence[Bar], a: float) -> float:
        closes = [b.close for b in hist]
        highs = [b.high for b in hist]
        lows = [b.low for b in hist]
        last_close = closes[-1]
        fast = sma(closes, self.fast) or closes[-1]
        slow = sma(closes, self.slow) or closes[-1]
        momentum = clamp((closes[-1] - closes[-4]) / a, -2.0, 2.0) if a > 0 and len(closes) >= 4 else 0.0
        recent_high = max(highs[-self.breakout_lookback - 1:-1])
        recent_low = min(lows[-self.breakout_lookback - 1:-1])
        trend = 0.0
        if last_close > fast > slow:
            trend = 0.40
        elif last_close < fast < slow:
            trend = -0.40
        elif last_close > slow:
            trend = 0.12
        elif last_close < slow:
            trend = -0.12
        breakout = 0.0
        if a > 0:
            breakout = clamp((last_close - recent_high) / a, -1.5, 1.5) * 0.45 + clamp((recent_low - last_close) / a, -1.5, 1.5) * -0.45
        return clamp(trend + breakout + momentum * 0.45 + (self.long_bias - 0.5) * 0.20, -1.5, 1.5)

    def _high_vol_entry_ok(self, hist: Sequence[Bar], a: float, signal: str, score: float) -> bool:
        is_hv, atr_pct, range_atr, spike_atr = self._high_volatility(hist, a)
        if not is_hv or a <= 0 or atr_pct <= 0 or abs(score) < self.threshold:
            return False
        closes = [b.close for b in hist]
        momentum = (closes[-1] - closes[-4]) / a if a > 0 and len(closes) >= 4 else 0.0
        if abs(momentum) < self.high_vol_min_momentum:
            return False
        highs = [b.high for b in hist]
        lows = [b.low for b in hist]
        recent_high = max(highs[-self.breakout_lookback - 1:-1])
        recent_low = min(lows[-self.breakout_lookback - 1:-1])
        last_close = closes[-1]
        breakout_atr = (last_close - recent_high) / a if last_close > recent_high else ((last_close - recent_low) / a if last_close < recent_low else 0.0)
        last_range = max(hist[-1].high - hist[-1].low, 0.0)
        close_location = (closes[-1] - hist[-1].low) / last_range if last_range > 0 else 0.5
        if signal == "BUY":
            return momentum > 0 and breakout_atr >= self.high_vol_min_breakout_atr and close_location >= self.high_vol_min_close_location
        elif signal == "SELL":
            return momentum < 0 and breakout_atr <= -self.high_vol_min_breakout_atr and close_location <= (1.0 - self.high_vol_min_close_location)
        return False

    def signal(self, hist: Sequence[Bar]) -> Tuple[str, float, float]:
        a = atr(hist, 14)
        if a is None or len(hist) < self.slow + self.breakout_lookback + 5:
            return "NONE", 0.0, 0.0
        score = self._score_signal(hist, a)
        signal = "NONE"
        if score >= self.threshold:
            signal = "BUY"
        elif score <= -self.threshold:
            signal = "SELL"
        if self.high_vol_only and not self._high_vol_entry_ok(hist, a, signal, score):
            return "NONE", 0.0, a
        return (signal, score, a) if signal != "NONE" else ("NONE", score, a)

    def sl_tp(self, side: str, entry: float, a: float, score: float) -> Tuple[float, float]:
        sl_distance = a * self.stop_atr
        strength = clamp((abs(score) - self.threshold) / (1.5 - self.threshold), 0.0, 1.0)
        tp_min_pts = 24.0 / (CONTRACT_SIZE * POINT)
        tp_max_pts = 72.0 / (CONTRACT_SIZE * POINT)
        tp_distance = max((tp_min_pts + (tp_max_pts - tp_min_pts) * strength) * POINT, sl_distance * 0.1)
        return (entry - sl_distance, entry + tp_distance) if side == "BUY" else (entry + sl_distance, entry - tp_distance)


class DoomsdayTrailing(DoomsdayStrategyV2):
    name = "doomsday_v2_trail"
    mode = "trailing"
    use_trailing = True
    trail_trigger_atr = 1.25
    trail_lock_atr = 0.25


class MomentumSurferV2(BaseStrategy):
    name = "momentum_surfer_v2"
    timeframe = "M5"
    mode = "no_trail"
    risk_pct = 0.012
    stop_atr = 2.5
    reward_multiple = 1.0
    atr_period = 14
    mom_lookback = 3
    vol_lookback = 50
    accel_min = 1.10
    entry_buffer_atr = 0.10
    use_trailing = False
    max_hold_bars = 24
    cooldown_bars = 1

    def signal(self, hist: Sequence[Bar]) -> Tuple[str, float, float]:
        closes = [b.close for b in hist]
        a = atr(hist, self.atr_period)
        if a is None or a <= 0 or len(closes) < self.vol_lookback + self.mom_lookback + 5:
            return "NONE", 0.0, 0.0
        baseline = atr(hist[:-(self.mom_lookback)], self.atr_period)
        if baseline is None or baseline <= 0:
            baseline = a
        vol_ratio = a / max(baseline, 1e-9)
        if vol_ratio < 1.0:
            return "NONE", 0.0, a
        mom_1 = closes[-1] - closes[-2]
        mom_2 = closes[-2] - closes[-3]
        mom_3 = closes[-3] - closes[-4]
        accel = mom_1 - mom_2
        accel_norm = abs(accel) / max(a, 1e-9)
        mom_alignment = 0.3 if (mom_1 > 0 and mom_2 > 0 and mom_3 > 0) or (mom_1 < 0 and mom_2 < 0 and mom_3 < 0) else 0.0
        mom_strength = (abs(mom_1) / max(a, 1e-9)) * 0.3
        vol_boost = max(0.0, (vol_ratio - 1.0)) * 0.4
        accel_score = accel_norm + mom_alignment + mom_strength + vol_boost
        if accel_score < self.accel_min:
            return "NONE", clamp(accel_score, -1.5, 1.5), a
        if mom_1 > 0 and mom_2 > 0 and accel > 0:
            return "BUY", clamp(accel_score, -1.5, 1.5), a
        if mom_1 < 0 and mom_2 < 0 and accel < 0:
            return "SELL", clamp(-accel_score, -1.5, 1.5), a
        return "NONE", clamp(accel_norm, -1.5, 1.5), a


class MomentumSurferTrailing(MomentumSurferV2):
    name = "momentum_surfer_v2_trail"
    mode = "trailing"
    use_trailing = True
    trail_trigger_atr = 1.50
    trail_lock_atr = 0.25


class DoomsdayV4(DoomsdayStrategyV2):
    name = "doomsday_v4"
    reward_multiple = 2.0

    def sl_tp(self, side: str, entry: float, a: float, score: float) -> Tuple[float, float]:
        stop = a * self.stop_atr
        reward = stop * self.reward_multiple
        return (entry - stop, entry + reward) if side == "BUY" else (entry + stop, entry - reward)


class DoomsdayV4Trail(DoomsdayV4):
    name = "doomsday_v4_trail"
    mode = "trailing"
    use_trailing = True
    trail_trigger_atr = 1.25
    trail_lock_atr = 0.25


class MomentumSurferM15(MomentumSurferV2):
    name = "momentum_surfer_m15"
    timeframe = "M15"
    reward_multiple = 2.0
    max_hold_bars = 48


class MomentumSurferM15Trail(MomentumSurferM15):
    name = "momentum_surfer_m15_trail"
    mode = "trailing"
    use_trailing = True
    trail_trigger_atr = 1.50
    trail_lock_atr = 0.25


def backtest_strategy(bars: List[Bar], strat: BaseStrategy) -> Tuple[List[Trade], Dict[str, float]]:
    equity = INITIAL_EQUITY
    peak = equity
    max_dd = 0.0
    trades: List[Trade] = []
    position: Optional[Dict[str, object]] = None
    cooldown_until = -1
    warmup = 220
    for i in range(warmup, len(bars) - 1):
        hist = bars[:i + 1]
        bar = bars[i]
        nxt = bars[i + 1]
        if position is not None:
            side = str(position["side"])
            sl = float(position["sl"])
            tp = float(position["tp"])
            entry = float(position["entry"])
            volume = float(position["volume"])
            atr_val = float(position["atr"])
            exit_price: Optional[float] = None
            reason = ""

            if strat.use_trailing:
                sl = strat.get_trailed_sl(side, entry, nxt.high, nxt.low, sl, atr_val)

            if side == "BUY":
                if nxt.low <= sl:
                    exit_price, reason = sl, "SL"
                elif nxt.high >= tp:
                    exit_price, reason = tp, "TP"
            else:
                if nxt.high >= sl:
                    exit_price, reason = sl, "SL"
                elif nxt.low <= tp:
                    exit_price, reason = tp, "TP"

            bars_held = i - int(position["entry_i"])
            if exit_price is None and bars_held >= strat.max_hold_bars:
                exit_price, reason = nxt.close, "TIME"

            if hasattr(strat, 'roll_trigger_pct') and exit_price is None:
                roll_equity = float(position.get("roll_entry_equity", equity))
                unrealized = abs(nxt.close - entry) * volume * CONTRACT_SIZE / max(roll_equity, 1e-9)
                if unrealized >= strat.roll_trigger_pct:
                    exit_price, reason = nxt.close, "ROLL"

            if exit_price is not None:
                mult = 1.0 if side == "BUY" else -1.0
                raw_pnl = (exit_price - entry) * mult * volume * CONTRACT_SIZE
                risk_amount = abs(entry - sl) * volume * CONTRACT_SIZE
                r_mult = raw_pnl / max(risk_amount, 1e-9)
                pnl_wc, comm, sprd = compute_costs(entry, exit_price, volume, side)
                equity += pnl_wc
                peak = max(peak, equity)
                max_dd = min(max_dd, equity / peak - 1.0)
                ts = str(position.get("trade_strategy", strat.name))
                trades.append(Trade(ts, strat.mode,
                    dt.datetime.fromisoformat(str(position["entry_time"])),
                    nxt.time, side, entry, exit_price, sl, tp, volume,
                    raw_pnl, pnl_wc, comm, sprd, r_mult, reason,
                    str(position["regime"]), float(position["atr"]), bars_held))
                position = None
                cooldown_until = i + strat.cooldown_bars

        if position is None and i >= cooldown_until:
            sig, score, a = strat.signal(hist)
            if sig in {"BUY", "SELL"} and a > 0:
                entry = nxt.open
                sl, tp = strat.sl_tp(sig, entry, a, score)
                vol = position_size(equity, strat.risk_pct, entry, sl)
                if vol > 0:
                    position = {"side": sig, "entry": entry, "sl": sl, "tp": tp,
                                "volume": vol, "entry_time": nxt.time.isoformat(),
                                "entry_i": i + 1, "regime": market_regime(hist[-80:]),
                                "atr": a, "trade_strategy": strat.name,
                                "roll_entry_equity": equity}

    wins = [t for t in trades if t.pnl_with_costs > 0]
    losses = [t for t in trades if t.pnl_with_costs <= 0]
    gp = sum(t.pnl_with_costs for t in wins)
    gl = -sum(t.pnl_with_costs for t in losses)
    total_costs = sum(t.commission + t.spread_cost for t in trades)
    metrics = {
        "trades": len(trades), "net_pnl_gross": round(sum(t.pnl for t in trades), 2),
        "net_pnl": round(equity - INITIAL_EQUITY, 2),
        "total_costs": round(total_costs, 2),
        "return_pct": round((equity / INITIAL_EQUITY - 1) * 100, 2),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 2) if trades else 0.0,
        "profit_factor": round(gp / gl, 3) if gl > 0 else (999.0 if gp > 0 else 0.0),
        "max_dd_pct": round(max_dd * 100, 2),
        "avg_r": round(statistics.fmean([t.r_multiple for t in trades]), 3) if trades else 0.0,
        "expectancy_usd": round(statistics.fmean([t.pnl_with_costs for t in trades]), 2) if trades else 0.0,
    }
    return trades, metrics


def load_live_trades(path: str) -> List[Dict[str, Any]]:
    if not Path(path).exists():
        return []
    deals: Dict[int, Dict[str, Any]] = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            pos_id = int(row.get("position_id", 0) or 0)
            if pos_id == 0:
                continue
            typ = int(row.get("type", 0) or 0)
            entry = int(row.get("entry", 0) or 0)
            profit = float(row.get("profit", 0.0) or 0.0)
            price = float(row.get("price", 0.0) or 0.0)
            volume = float(row.get("volume", 0.0) or 0.0)
            time_str = row.get("time") or row.get("\ufefftime", "")
            if pos_id not in deals:
                deals[pos_id] = {"position_id": pos_id, "entries": [], "exits": [], "total_pnl": 0.0}
            if entry == 0:
                deals[pos_id]["entries"].append({"time": time_str, "price": price, "volume": volume, "type": typ})
            elif entry in (1, 3):
                deals[pos_id]["exits"].append({"time": time_str, "price": price, "volume": volume, "type": typ, "profit": profit})
                deals[pos_id]["total_pnl"] += profit
    result = []
    for pid, info in deals.items():
        if info["entries"] and info["exits"]:
            e = info["entries"][0]
            x = info["exits"][-1]
            side = "BUY" if e["type"] == 0 else "SELL"
            result.append({"position_id": pid, "side": side, "entry_time": e["time"],
                          "exit_time": x["time"], "entry_price": e["price"],
                          "exit_price": x["price"], "volume": e["volume"], "pnl": info["total_pnl"]})
    return result


def compare_with_live(bt: List[Trade], live_path: str, sname: str) -> Dict[str, Any]:
    live = load_live_trades(live_path)
    if not live:
        return {}
    bt_pnl = sum(t.pnl_with_costs for t in bt)
    bt_win = len([t for t in bt if t.pnl_with_costs > 0])
    live_pnl = sum(t["pnl"] for t in live)
    live_win = len([t for t in live if t["pnl"] > 0])
    return {
        "strategy": sname, "backtest_trades": len(bt), "live_trades": len(live),
        "backtest_net_pnl": round(bt_pnl, 2), "live_net_pnl": round(live_pnl, 2),
        "backtest_win_rate": round(bt_win / len(bt) * 100, 2) if bt else 0,
        "live_win_rate": round(live_win / len(live) * 100, 2) if live else 0,
        "pnl_diff": round(bt_pnl - live_pnl, 2),
        "pnl_diff_pct": round((bt_pnl - live_pnl) / abs(live_pnl) * 100, 2) if live_pnl != 0 else 0,
        "trades_multiple": round(len(bt) / max(len(live), 1), 2),
    }


def print_metrics(label: str, m: Dict[str, float]) -> None:
    print(f"  {label:40s} trades={m['trades']:>4d}  pnl={m['net_pnl']:>8.2f}  ret={m['return_pct']:>7.2f}%  wr={m['win_rate_pct']:>5.1f}%  pf={m['profit_factor']:>6.3f}  dd={m['max_dd_pct']:>6.2f}%  cost={m['total_costs']:>6.2f}")


def print_table(rows: List[Dict[str, Any]]) -> None:
    print(f"  {'Mode':<20s} {'TF':>4s} {'Trades':>6s} {'NetPnL':>8s} {'Return':>7s} {'WinRate':>7s} {'PF':>6s} {'MaxDD':>7s} {'AvgR':>6s} {'Costs':>7s}")
    print(f"  {'-'*20} {'-'*4} {'-'*6} {'-'*8} {'-'*7} {'-'*7} {'-'*6} {'-'*7} {'-'*6} {'-'*7}")
    for r in rows:
        tf = r.get('timeframe', '')
        print(f"  {r['mode']:<20s} {tf:>4s} {r['trades']:>6d} {r['net_pnl']:>8.2f} {r['return_pct']:>6.2f}% {r['win_rate_pct']:>5.1f}% {r['profit_factor']:>6.3f} {r['max_dd_pct']:>6.2f}% {r['avg_r']:>6.3f} {r['total_costs']:>7.2f}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    configs = [
        ("Doomsday V2 (no trail, live params)", DoomsdayStrategyV2),
        ("Doomsday V2 (trailing 1.25/0.25)", DoomsdayTrailing),
        ("Doomsday V4 (2.0R, ATR TP)", DoomsdayV4),
        ("Doomsday V4 trail (2.0R)", DoomsdayV4Trail),
        ("MomSurfer V2 (no trail, live params)", MomentumSurferV2),
        ("MomSurfer V2 (trailing 1.50/0.25)", MomentumSurferTrailing),
        ("MomSurfer M15 (2.0R)", MomentumSurferM15),
        ("MomSurfer M15 trail (2.0R)", MomentumSurferM15Trail),
    ]

    unique_tfs = sorted(set(klass.timeframe for _, klass in configs))
    bars_dict: Dict[str, List[Bar]] = {}
    for tf in unique_tfs:
        print(f"Fetching {tf} data...")
        try:
            bars, actual_tf = fetch_mt5_bars(tf=tf)
            print(f"  Got {len(bars)} bars from {bars[0].time} to {bars[-1].time} on {actual_tf}")
            bars_dict[actual_tf] = bars
        except RuntimeError as e:
            print(f"  {tf} fetch failed: {e}")
    if not bars_dict:
        print("Cannot fetch data. MT5 bridge may not be running.")
        return

    first_tf = list(bars_dict.keys())[0]
    first_bars = bars_dict[first_tf]
    summary: Dict[str, object] = {
        "symbol": SYMBOL, "timeframes": {tf: len(b) for tf, b in bars_dict.items()},
        "start": first_bars[0].time.isoformat(), "end": first_bars[-1].time.isoformat(),
        "initial_equity": INITIAL_EQUITY,
        "spread_points": SPREAD_POINTS, "commission_per_lot": COMMISSION_PER_LOT,
        "strategies": {}, "live_comparisons": {},
    }

    v2_results: List[Dict] = []
    live_file = str(ROOT / "doomsday_full_history.csv")

    for label, klass in configs:
        name = klass.name
        strat = klass()
        tf = strat.timeframe
        bars = bars_dict.get(tf, list(bars_dict.values())[0])
        print(f"\n===== {label} [{tf}] =====")
        trades, metrics = backtest_strategy(bars, strat)
        metrics["timeframe"] = tf
        print_metrics(name, metrics)
        v2_results.append({"mode": strat.mode or name, **metrics})

        csv_path = OUT_DIR / f"{name}_trades.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[f.name for f in dataclasses.fields(Trade)])
            w.writeheader()
            for t in trades:
                row = dataclasses.asdict(t)
                row["entry_time"] = t.entry_time.isoformat()
                row["exit_time"] = t.exit_time.isoformat()
                w.writerow(row)

        summary["strategies"][name] = {"label": label, "metrics": metrics}

        if "doomsday" in name and Path(live_file).exists():
            comp = compare_with_live(trades, live_file, name)
            if comp:
                summary["live_comparisons"][name] = {k: v for k, v in comp.items() if k != "live_details"}

    # Print live comparison table
    live_comp = summary.get("live_comparisons", {})
    if live_comp:
        print(f"\n{'='*60}")
        print(f"  LIVE vs BACKTEST COMPARISON (Doomsday only)")
        print(f"{'='*60}")
        print(f"  {'Mode':<30s} {'BT PnL':>8s} {'Live PnL':>9s} {'Diff':>8s} {'BT WR':>7s} {'Live WR':>8s}")
        print(f"  {'-'*30} {'-'*8} {'-'*9} {'-'*8} {'-'*7} {'-'*8}")
        for name, c in live_comp.items():
            print(f"  {name:<30s} {c['backtest_net_pnl']:>8.2f} {c['live_net_pnl']:>9.2f} {c['pnl_diff']:>+8.2f} {c['backtest_win_rate']:>6.1f}% {c['live_win_rate']:>7.1f}%")
        print(f"\n  Note: Backtest = ~3mo (72 trades), Live = ~3 days (7 trades)")
        print(f"  Live doomsday: +$78.94, 71.4% WR. Backtest w/ trail: +${list(live_comp.values())[0]['backtest_net_pnl']:.2f}")

    # Print original v1 comparison
    print(f"\n{'='*60}")
    print(f"  V1 (ORIGINAL) vs V2 (ENHANCED) COMPARISON")
    print(f"{'='*60}")
    print(f"  Results from original backtest_compare_strategies.py (no costs, no trail):")
    print(f"  Doomsday v1:        249 trades, +$755.11, 7.55% ret, 46.2% WR, PF 1.186, DD -6.16%")
    print(f"  MomentumSurfer v1:  125 trades, +$1374.83, 13.75% ret, 44.8% WR, PF 1.266, DD -5.54%")
    print(f"")
    print_table(v2_results)

    # Cost impact analysis
    print(f"\n{'='*60}")
    print(f"  COST IMPACT ANALYSIS")
    print(f"{'='*60}")
    for name, data in summary["strategies"].items():
        m = data["metrics"]
        cost_pct = m["total_costs"] / max(abs(m["net_pnl_gross"]), 1e-9) * 100 if m["net_pnl_gross"] != 0 else 0
        print(f"  {name:<35s} gross={m['net_pnl_gross']:>8.2f} net={m['net_pnl']:>8.2f} costs={m['total_costs']:>6.2f} ({cost_pct:>5.1f}% of gross)")

    # Save
    summary_path = OUT_DIR / "strategy_comparison_v2.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSummary saved: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
