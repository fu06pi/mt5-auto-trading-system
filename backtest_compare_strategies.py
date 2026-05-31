#!/usr/bin/env python3
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from pymt5linux import MetaTrader5
except ImportError:  # allow running with system python by falling back to known venv
    import sys
    sys.path.insert(0, "/home/chain4655/.openharness-venv/lib64/python3.14/site-packages")
    from pymt5linux import MetaTrader5

ROOT = Path("/home/chain4655/Documents/Projects/MT5")
OUT_DIR = ROOT / "backtest_reports"
SYMBOL = "XAUUSD"
TIMEFRAME = "M15"
TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
INITIAL_EQUITY = 10000.0
CONTRACT_SIZE = 100.0
POINT = 0.01


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
    entry_time: dt.datetime
    exit_time: dt.datetime
    side: str
    entry: float
    exit: float
    sl: float
    tp: float
    volume: float
    pnl: float
    r_multiple: float
    reason: str
    regime: str
    atr: float
    bars_held: int


def fetch_mt5_bars(months: int = 3) -> List[Bar]:
    mt5 = MetaTrader5(host="127.0.0.1", port=18812)
    ok = mt5.initialize(path=TERMINAL_PATH)
    if not ok:
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        end = dt.datetime.now()
        start = end - dt.timedelta(days=92)
        rates = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M15, start, end)
        if rates is None or len(rates) == 0:
            # 3 months M15 ~= 96 bars/day * 92 days
            rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, 9000)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No rates returned: {mt5.last_error()}")
        bars: List[Bar] = []
        for row in rates:
            # numpy structured array row
            ts = int(row["time"])
            bars.append(
                Bar(
                    time=dt.datetime.fromtimestamp(ts),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    tick_volume=float(row["tick_volume"]),
                )
            )
        bars.sort(key=lambda b: b.time)
        return bars
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


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


class BaseStrategy:
    name = "base"
    risk_pct = 0.0075
    stop_atr = 5.0
    reward_multiple = 2.2
    max_hold_bars = 32
    cooldown_bars = 4

    def signal(self, hist: Sequence[Bar]) -> Tuple[str, float, float]:
        return "NONE", 0.0, 0.0

    def sl_tp(self, side: str, entry: float, a: float, score: float) -> Tuple[float, float]:
        stop = a * self.stop_atr
        reward = stop * self.reward_multiple
        if side == "BUY":
            return entry - stop, entry + reward
        return entry + stop, entry - reward


class DoomsdayStrategy(BaseStrategy):
    name = "doomsday"
    risk_pct = 0.006
    stop_atr = 3.8
    reward_multiple = 1.6
    long_bias = 0.70
    threshold = 0.58
    fast = 8
    slow = 34
    breakout_lookback = 18
    high_vol_atr_pct = 0.0018
    high_vol_range_atr = 4.0
    high_vol_min_momentum = 0.80
    max_hold_bars = 18

    def _high_volatility(self, hist: Sequence[Bar], a: float) -> Tuple[bool, float, float]:
        closes = [b.close for b in hist]
        highs = [b.high for b in hist]
        lows = [b.low for b in hist]
        atr_pct = a / max(closes[-1], 1e-9)
        lookback = min(20, len(highs), len(lows))
        range_atr = (max(highs[-lookback:]) - min(lows[-lookback:])) / max(a, 1e-9)
        return atr_pct >= self.high_vol_atr_pct or range_atr >= self.high_vol_range_atr, atr_pct, range_atr

    def signal(self, hist: Sequence[Bar]) -> Tuple[str, float, float]:
        closes = [b.close for b in hist]
        highs = [b.high for b in hist]
        lows = [b.low for b in hist]
        a = atr(hist, 18)
        if a is None or len(closes) < self.slow + self.breakout_lookback + 5:
            return "NONE", 0.0, 0.0
        is_high_vol, _atr_pct, _range_atr = self._high_volatility(hist, a)
        if not is_high_vol:
            return "NONE", 0.0, a
        fast = sma(closes, self.fast) or closes[-1]
        slow = sma(closes, self.slow) or closes[-1]
        momentum = clamp((closes[-1] - closes[-4]) / a, -2.0, 2.0)
        if abs(momentum) < self.high_vol_min_momentum:
            return "NONE", 0.0, a
        last_close = closes[-1]
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
        breakout = clamp((last_close - recent_high) / a, -1.5, 1.5) * 0.45
        breakout += clamp((recent_low - last_close) / a, -1.5, 1.5) * -0.45
        score = clamp(trend + breakout + momentum * 0.45 + (self.long_bias - 0.5) * 0.20, -1.5, 1.5)
        if score >= self.threshold:
            return "BUY", score, a
        if score <= -self.threshold:
            return "SELL", score, a
        return "NONE", score, a


class BollingerEdgeSqueezeStrategy(BaseStrategy):
    name = "bollinger_edge_squeeze"
    risk_pct = 0.007
    stop_atr = 5.5
    reward_multiple = 2.2
    threshold = 0.48
    bb_period = 20
    bb_stddev = 2.0
    edge_pct = 0.16
    squeeze_lookback = 90
    squeeze_quantile = 0.25
    expansion_ratio = 1.25
    max_hold_bars = 28

    def _bands(self, closes: Sequence[float]) -> Optional[Dict[str, float]]:
        if len(closes) < self.bb_period:
            return None
        win = list(closes[-self.bb_period:])
        mid = statistics.fmean(win)
        sig = statistics.pstdev(win) if len(win) > 1 else 0.0
        upper = mid + sig * self.bb_stddev
        lower = mid - sig * self.bb_stddev
        bw = max(upper - lower, POINT)
        return {"mid": mid, "upper": upper, "lower": lower, "position": (closes[-1] - lower) / bw, "bandwidth_pct": bw / max(abs(mid), POINT)}

    def signal(self, hist: Sequence[Bar]) -> Tuple[str, float, float]:
        lookback = self.bb_period + self.squeeze_lookback + 10
        closes = [b.close for b in hist[-lookback:]]
        a = atr(hist, 10)
        if a is None or len(closes) < self.bb_period + self.squeeze_lookback + 2:
            return "NONE", 0.0, 0.0
        bands = self._bands(closes)
        if bands is None:
            return "NONE", 0.0, a
        history = []
        for end in range(self.bb_period, len(closes) + 1):
            b = self._bands(closes[:end])
            if b:
                history.append(b["bandwidth_pct"])
        recent = history[-self.squeeze_lookback:]
        squeeze_threshold = percentile(recent, self.squeeze_quantile)
        previous_bw = history[-2] if len(history) >= 2 else bands["bandwidth_pct"]
        bandwidth_ratio = bands["bandwidth_pct"] / max(previous_bw, 1e-9)
        squeeze_active = bands["bandwidth_pct"] <= squeeze_threshold
        recent_squeeze = any(bw <= squeeze_threshold for bw in history[-7:-1])
        squeeze_release = recent_squeeze and bandwidth_ratio >= self.expansion_ratio
        direction = 0.0
        if bands["position"] >= 1.0 - self.edge_pct:
            direction = 1.0
        elif bands["position"] <= self.edge_pct:
            direction = -1.0
        momentum = clamp((closes[-1] - closes[-4]) / a, -2.0, 2.0)
        score = direction * 0.55 + momentum * 0.20
        if squeeze_release:
            score += math.copysign(0.35, momentum if momentum != 0 else direction)
        if squeeze_active:
            score *= 0.55
        score = clamp(score, -1.5, 1.5)
        if score >= self.threshold:
            return "BUY", score, a
        if score <= -self.threshold:
            return "SELL", score, a
        return "NONE", score, a


class TrendStrategy(BaseStrategy):
    name = "trend_following"
    risk_pct = 0.006
    stop_atr = 3.0
    reward_multiple = 2.5
    threshold = 0.85
    fast = 28
    slow = 50
    breakout_lookback = 30
    max_hold_bars = 48

    def signal(self, hist: Sequence[Bar]) -> Tuple[str, float, float]:
        closes = [b.close for b in hist]
        highs = [b.high for b in hist]
        lows = [b.low for b in hist]
        a = atr(hist, 14)
        if a is None or len(closes) < self.slow + 5:
            return "NONE", 0.0, 0.0
        fast = sma(closes, self.fast) or closes[-1]
        slow = sma(closes, self.slow) or closes[-1]
        momentum = clamp((closes[-1] - closes[-5]) / a, -2.0, 2.0)
        breakout_up = closes[-1] > max(highs[-self.breakout_lookback:-1])
        breakout_dn = closes[-1] < min(lows[-self.breakout_lookback:-1])
        score = 0.0
        if closes[-1] > fast > slow:
            score += 0.65
        elif closes[-1] < fast < slow:
            score -= 0.65
        score += momentum * 0.25
        if breakout_up:
            score += 0.35
        if breakout_dn:
            score -= 0.35
        score = clamp(score, -1.5, 1.5)
        if score >= self.threshold:
            return "BUY", score, a
        if score <= -self.threshold:
            return "SELL", score, a
        return "NONE", score, a


class AsiaLondonBreakoutStrategy(BaseStrategy):
    name = "asia_london_breakout_ep1"
    risk_pct = 0.006
    stop_atr = 4.4
    reward_multiple = 2.4
    asia_start = 0
    asia_end = 6
    london_start = 7
    london_end = 16
    buffer_atr = 0.21
    min_asia_atr = 0.55
    max_asia_atr = 3.0
    max_hold_bars = 20

    def signal(self, hist: Sequence[Bar]) -> Tuple[str, float, float]:
        a = atr(hist, 24)
        if a is None or len(hist) < 120:
            return "NONE", 0.0, 0.0
        last = hist[-1]
        hour = last.time.hour
        if not (self.london_start <= hour < self.london_end):
            return "NONE", 0.0, a
        day = last.time.date()
        asia = [b for b in hist if b.time.date() == day and self.asia_start <= b.time.hour < self.asia_end]
        if len(asia) < 8:
            return "NONE", 0.0, a
        ah = max(b.high for b in asia)
        al = min(b.low for b in asia)
        width_atr = (ah - al) / max(a, 1e-9)
        if width_atr < self.min_asia_atr or width_atr > self.max_asia_atr:
            return "NONE", 0.0, a
        buffer = a * self.buffer_atr
        # HTF alignment approximation: last 55/110 SMA
        closes = [b.close for b in hist]
        htf_fast = sma(closes, 55) or closes[-1]
        htf_slow = sma(closes, 110) or closes[-1]
        score = 0.0
        if last.close > ah + buffer and htf_fast >= htf_slow:
            score = 1.0
            return "BUY", score, a
        if last.close < al - buffer and htf_fast <= htf_slow:
            score = -1.0
            return "SELL", score, a
        return "NONE", 0.0, a


class MomentumSurferStrategy(BaseStrategy):
    name = "momentum_surfer"
    risk_pct = 0.01
    stop_atr = 2.5
    reward_multiple = 3.0
    threshold = 0.80
    atr_period = 14
    mom_lookback = 3
    vol_lookback = 50
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
        mom_1_norm = abs(mom_1) / max(a, 1e-9)
        vol_boost = max(0.0, vol_ratio - 1.0)

        strength = accel_norm * 0.50 + mom_1_norm * 0.30 + vol_boost * 0.20

        if mom_1 > 0 and mom_2 > 0 and accel > 0 and strength >= self.threshold:
            return "BUY", clamp(strength, -1.5, 1.5), a
        if mom_1 < 0 and mom_2 < 0 and accel < 0 and strength >= self.threshold:
            return "SELL", clamp(-strength, -1.5, 1.5), a

        return "NONE", clamp(accel_norm, -1.5, 1.5), a


class MetaRegimeSwitchStrategy(BaseStrategy):
    name = "meta_regime_switch"
    cooldown_bars = 4
    max_hold_bars = 40

    def __init__(self) -> None:
        self.doomsday = DoomsdayStrategy()
        self.bollinger = BollingerEdgeSqueezeStrategy()
        self.trend = TrendStrategy()
        self.ep1 = AsiaLondonBreakoutStrategy()
        self.selected_name = "none"

    @property
    def risk_pct(self) -> float:  # set dynamically after signal selection
        return getattr(self, "_risk_pct", 0.005)

    @risk_pct.setter
    def risk_pct(self, value: float) -> None:
        self._risk_pct = value

    def sl_tp(self, side: str, entry: float, a: float, score: float) -> Tuple[float, float]:
        return self._selected.sl_tp(side, entry, a, score)

    def _is_compression_release(self, hist: Sequence[Bar]) -> bool:
        # Use only the required rolling window; using the full 3-month history
        # inside every bar makes the meta backtest quadratic and unnecessarily slow.
        lookback = self.bollinger.bb_period + self.bollinger.squeeze_lookback + 10
        closes = [b.close for b in hist[-lookback:]]
        if len(closes) < self.bollinger.bb_period + self.bollinger.squeeze_lookback + 2:
            return False
        b = self.bollinger._bands(closes)
        if not b:
            return False
        history = []
        for end in range(self.bollinger.bb_period, len(closes) + 1):
            bands = self.bollinger._bands(closes[:end])
            if bands:
                history.append(bands["bandwidth_pct"])
        recent = history[-self.bollinger.squeeze_lookback:]
        threshold = percentile(recent, self.bollinger.squeeze_quantile)
        previous = history[-2] if len(history) >= 2 else b["bandwidth_pct"]
        ratio = b["bandwidth_pct"] / max(previous, 1e-9)
        recent_squeeze = any(bw <= threshold for bw in history[-7:-1])
        return bool(recent_squeeze and ratio >= self.bollinger.expansion_ratio)

    def signal(self, hist: Sequence[Bar]) -> Tuple[str, float, float]:
        regime = market_regime(hist[-80:])
        last = hist[-1]
        hour = last.time.hour
        candidates: List[Tuple[str, BaseStrategy, str, float, float]] = []

        # London open gets first right of refusal, but only if it actually fires.
        if 7 <= hour < 16:
            sig, score, a = self.ep1.signal(hist)
            if sig != "NONE":
                candidates.append(("asia_london_breakout_ep1", self.ep1, sig, score, a))

        # Compression-release / mixed/high-volatility: Bollinger only when release actually exists.
        if regime in {"range/compression", "mixed", "high-volatility"} and self._is_compression_release(hist):
            sig, score, a = self.bollinger.signal(hist)
            if sig != "NONE":
                candidates.append(("bollinger_edge_squeeze", self.bollinger, sig, score, a))

        # Mixed: trend_following had the best tested edge, doomsday only if high-conviction.
        if regime == "mixed":
            sig, score, a = self.trend.signal(hist)
            if sig != "NONE":
                candidates.append(("trend_following", self.trend, sig, score, a))
            sig, score, a = self.doomsday.signal(hist)
            if sig != "NONE" and abs(score) >= 0.75:
                candidates.append(("doomsday", self.doomsday, sig, score, a))

        # Trend: only allow stronger trend signals, avoid doomsday/Bollinger trend chop.
        if regime == "trend":
            sig, score, a = self.trend.signal(hist)
            if sig != "NONE" and abs(score) >= 0.9:
                candidates.append(("trend_following", self.trend, sig, score, a))

        # High volatility: doomsday is now the dedicated high-volatility momentum module.
        if regime == "high-volatility":
            sig, score, a = self.doomsday.signal(hist)
            if sig != "NONE" and abs(score) >= 0.75:
                candidates.append(("doomsday", self.doomsday, sig, score, a))

        if not candidates:
            self.selected_name = "none"
            return "NONE", 0.0, atr(hist, 14) or 0.0

        # Prefer the strongest absolute score among allowed regime candidates.
        name, strat, sig, score, a = max(candidates, key=lambda x: abs(x[3]))
        self.selected_name = name
        self._selected = strat
        self.risk_pct = min(strat.risk_pct, 0.006 if regime == "high-volatility" else strat.risk_pct)
        self.stop_atr = strat.stop_atr
        self.reward_multiple = strat.reward_multiple
        self.max_hold_bars = strat.max_hold_bars
        return sig, score, a


def position_size(equity: float, risk_pct: float, entry: float, sl: float, max_lots: float = 5.0) -> float:
    risk = equity * risk_pct
    risk_per_lot = abs(entry - sl) * CONTRACT_SIZE
    if risk_per_lot <= 0:
        return 0.0
    lots = risk / risk_per_lot
    return max(0.01, min(max_lots, math.floor(lots / 0.01) * 0.01))


def backtest_strategy(bars: List[Bar], strat: BaseStrategy) -> Tuple[List[Trade], Dict[str, float], List[Dict[str, object]]]:
    equity = INITIAL_EQUITY
    peak = equity
    max_dd = 0.0
    trades: List[Trade] = []
    equity_curve: List[Dict[str, object]] = []
    position: Optional[Dict[str, object]] = None
    cooldown_until = -1
    warmup = 220
    for i in range(warmup, len(bars) - 1):
        hist = bars[: i + 1]
        bar = bars[i]
        nxt = bars[i + 1]
        if position is not None:
            side = str(position["side"])
            sl = float(position["sl"])
            tp = float(position["tp"])
            exit_price: Optional[float] = None
            reason = ""
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
            if exit_price is not None:
                entry = float(position["entry"])
                volume = float(position["volume"])
                mult = 1.0 if side == "BUY" else -1.0
                pnl = (exit_price - entry) * mult * volume * CONTRACT_SIZE
                risk_amount = abs(entry - sl) * volume * CONTRACT_SIZE
                r_mult = pnl / max(risk_amount, 1e-9)
                equity += pnl
                peak = max(peak, equity)
                max_dd = min(max_dd, equity / peak - 1.0)
                trade_strategy = str(position.get("trade_strategy", strat.name))
                trades.append(Trade(trade_strategy, dt.datetime.fromisoformat(str(position["entry_time"])), nxt.time, side, entry, exit_price, sl, tp, volume, pnl, r_mult, reason, str(position["regime"]), float(position["atr"]), bars_held))
                position = None
                cooldown_until = i + strat.cooldown_bars
        if position is None and i >= cooldown_until:
            sig, score, a = strat.signal(hist)
            if sig in {"BUY", "SELL"} and a > 0:
                entry = nxt.open
                sl, tp = strat.sl_tp(sig, entry, a, score)
                vol = position_size(equity, strat.risk_pct, entry, sl)
                if vol > 0:
                    selected_name = getattr(strat, "selected_name", strat.name)
                    trade_strategy = strat.name if strat.name != "meta_regime_switch" else f"meta::{selected_name}"
                    position = {
                        "side": sig,
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "volume": vol,
                        "entry_time": nxt.time.isoformat(),
                        "entry_i": i + 1,
                        "regime": market_regime(hist[-80:]),
                        "atr": a,
                        "trade_strategy": trade_strategy,
                    }
        equity_curve.append({"time": bar.time.isoformat(), "equity": equity, "peak": peak, "dd": equity / peak - 1.0})
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = -sum(t.pnl for t in losses)
    metrics = {
        "trades": len(trades),
        "net_pnl": round(equity - INITIAL_EQUITY, 2),
        "return_pct": round((equity / INITIAL_EQUITY - 1) * 100, 2),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 2) if trades else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0),
        "max_dd_pct": round(max_dd * 100, 2),
        "avg_r": round(statistics.fmean([t.r_multiple for t in trades]), 3) if trades else 0.0,
        "expectancy_usd": round(statistics.fmean([t.pnl for t in trades]), 2) if trades else 0.0,
    }
    return trades, metrics, equity_curve


def regime_stats(trades: Sequence[Trade]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for regime in sorted(set(t.regime for t in trades)):
        group = [t for t in trades if t.regime == regime]
        wins = [t for t in group if t.pnl > 0]
        gp = sum(t.pnl for t in wins)
        gl = -sum(t.pnl for t in group if t.pnl <= 0)
        out[regime] = {
            "trades": len(group),
            "net_pnl": round(sum(t.pnl for t in group), 2),
            "win_rate_pct": round(len(wins) / len(group) * 100, 2) if group else 0,
            "profit_factor": round(gp / gl, 3) if gl > 0 else (999.0 if gp > 0 else 0.0),
            "avg_r": round(statistics.fmean([t.r_multiple for t in group]), 3) if group else 0,
        }
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bars = fetch_mt5_bars()
    if len(bars) < 500:
        raise RuntimeError(f"Not enough bars: {len(bars)}")
    strategies: List[BaseStrategy] = [DoomsdayStrategy(), BollingerEdgeSqueezeStrategy(), TrendStrategy(), AsiaLondonBreakoutStrategy(), MomentumSurferStrategy(), MetaRegimeSwitchStrategy()]
    summary: Dict[str, object] = {
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "start": bars[0].time.isoformat(),
        "end": bars[-1].time.isoformat(),
        "bars": len(bars),
        "initial_equity": INITIAL_EQUITY,
        "strategies": {},
    }
    all_trades: List[Trade] = []
    for strat in strategies:
        trades, metrics, curve = backtest_strategy(bars, strat)
        all_trades.extend(trades)
        summary["strategies"][strat.name] = {"metrics": metrics, "regimes": regime_stats(trades)}
        with (OUT_DIR / f"{strat.name}_trades.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[field.name for field in dataclasses.fields(Trade)])
            writer.writeheader()
            for t in trades:
                row = dataclasses.asdict(t)
                row["entry_time"] = t.entry_time.isoformat()
                row["exit_time"] = t.exit_time.isoformat()
                writer.writerow(row)
    with (OUT_DIR / "strategy_comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (OUT_DIR / "strategy_comparison_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["strategy", "trades", "net_pnl", "return_pct", "win_rate_pct", "profit_factor", "max_dd_pct", "avg_r", "expectancy_usd"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, data in summary["strategies"].items():
            row = {"strategy": name}
            row.update(data["metrics"])
            writer.writerow(row)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
