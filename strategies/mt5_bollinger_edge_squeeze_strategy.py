"""MT5 Bollinger-band edge and squeeze strategy.

This strategy keeps the doomsday MT5 operating architecture while replacing
the directional signal model with Bollinger-band edge, squeeze, and bandwidth
expansion logic. It does not assume or promise the same performance profile as
the original strategy.

Run example:
    python3.14 /home/chain4655/Documents/Projects/MT5/strategies/mt5_bollinger_edge_squeeze_strategy.py --symbol XAUUSD --timeframe M5 --host 127.0.0.1 --port 18812 --live
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import logging
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence

from mt5_doomsday_strategy import (
    DoomsdayMT5Strategy,
    MarketSnapshot,
    PositionState,
    StrategyConfig as DoomsdayStrategyConfig,
    build_parser as build_doomsday_parser,
)


LOGGER = logging.getLogger("mt5_bollinger_edge_squeeze_strategy")
LOG_FILE_DEFAULT = "/home/chain4655/Documents/Projects/MT5/mt5_bollinger_edge_squeeze_strategy_supervised.log"
OPTIMIZATION_SPACE = {
    "risk_pct": {"type": "float", "min": 0.004, "max": 0.012, "step": 0.001, "default": 0.007},
    "stop_atr": {"type": "float", "min": 3.5, "max": 7.5, "step": 0.1, "default": 4.5},
    "reward_multiple": {"type": "float", "min": 5.0, "max": 12.0, "step": 0.25, "default": 7.5},
    "tp_min_usd": {"type": "float", "min": 20.0, "max": 60.0, "step": 2.5, "default": 36.0},
    "tp_max_usd": {"type": "float", "min": 50.0, "max": 120.0, "step": 5.0, "default": 75.0},
    "long_bias": {"type": "float", "min": 0.35, "max": 0.75, "step": 0.01, "default": 0.55},
    "trend_threshold": {"type": "float", "min": 0.08, "max": 0.45, "step": 0.01, "default": 0.30},
    "roll_trigger_pct": {"type": "float", "min": 0.04, "max": 0.18, "step": 0.01, "default": 0.08},
    "cooldown_minutes": {"type": "int", "min": 20, "max": 120, "step": 5, "default": 60},
    "max_leverage": {"type": "float", "min": 2.0, "max": 8.0, "step": 0.5, "default": 3.0},
    "max_drawdown_pct": {"type": "float", "min": 0.04, "max": 0.12, "step": 0.005, "default": 0.06},
    "max_daily_loss_pct": {"type": "float", "min": 0.02, "max": 0.05, "step": 0.005, "default": 0.03},
    "max_lots": {"type": "float", "min": 0.5, "max": 5.0, "step": 0.1, "default": 3.0},
    "deviation": {"type": "int", "min": 10, "max": 80, "step": 5, "default": 50},
    "loop_seconds": {"type": "int", "min": 5, "max": 30, "step": 1, "default": 9},
    "lookback_bars": {"type": "int", "min": 80, "max": 240, "step": 10, "default": 160},
    "atr_period": {"type": "int", "min": 8, "max": 24, "step": 1, "default": 10},
    "fast_sma": {"type": "int", "min": 5, "max": 24, "step": 1, "default": 13},
    "slow_sma": {"type": "int", "min": 18, "max": 60, "step": 2, "default": 28},
    "bb_period": {"type": "int", "min": 14, "max": 40, "step": 1, "default": 20},
    "bb_stddev": {"type": "float", "min": 1.4, "max": 3.0, "step": 0.1, "default": 2.0},
    "bb_edge_pct": {"type": "float", "min": 0.08, "max": 0.25, "step": 0.01, "default": 0.16},
    "squeeze_lookback": {"type": "int", "min": 30, "max": 180, "step": 5, "default": 90},
    "squeeze_quantile": {"type": "float", "min": 0.10, "max": 0.45, "step": 0.01, "default": 0.25},
    "expansion_ratio": {"type": "float", "min": 1.02, "max": 1.35, "step": 0.01, "default": 1.12},
    "min_bandwidth_atr": {"type": "float", "min": 0.15, "max": 0.80, "step": 0.01, "default": 0.35},
    "squeeze_release_bars": {"type": "int", "min": 2, "max": 12, "step": 1, "default": 6},
}


@dataclasses.dataclass(frozen=True)
class BollingerStrategyConfig(DoomsdayStrategyConfig):
    bb_period: int
    bb_stddev: float
    bb_edge_pct: float
    squeeze_lookback: int
    squeeze_quantile: float
    expansion_ratio: float
    min_bandwidth_atr: float
    squeeze_release_bars: int
    log_file: str
    state_path: Optional[str]

    @classmethod
    def optimization_space(cls) -> Dict[str, Dict[str, Any]]:
        return OPTIMIZATION_SPACE

    @classmethod
    def default_cli_args(cls) -> Dict[str, Any]:
        return {key: spec["default"] for key, spec in OPTIMIZATION_SPACE.items()}

    @classmethod
    def optimization_fields(cls) -> List[str]:
        return list(OPTIMIZATION_SPACE.keys())

    @classmethod
    def optimization_space_json(cls) -> str:
        import json

        return json.dumps(OPTIMIZATION_SPACE, indent=2, ensure_ascii=False)

    @classmethod
    def optimization_summary(cls) -> str:
        lines = ["Bollinger edge/squeeze optimization space:"]
        for key, spec in OPTIMIZATION_SPACE.items():
            lines.append(
                f"- {key}: {spec['type']} [{spec['min']}, {spec['max']}] step={spec['step']} default={spec['default']}"
            )
        return "\n".join(lines)


@dataclasses.dataclass
class BollingerSnapshot(MarketSnapshot):
    bb_mid: float = 0.0
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    bb_position: float = 0.5
    bandwidth: float = 0.0
    bandwidth_pct: float = 0.0
    bandwidth_atr: float = 0.0
    bandwidth_ratio: float = 1.0
    squeeze_threshold: float = 0.0
    squeeze_active: bool = False
    squeeze_release: bool = False
    edge: str = "NONE"


class BollingerEdgeSqueezeMT5Strategy(DoomsdayMT5Strategy):
    config: BollingerStrategyConfig

    def _log_file(self, message: str, *args: Any) -> None:
        try:
            with open(self.config.log_file, "a", encoding="utf-8") as f:
                f.write((message % args) + "\n")
        except OSError as exc:
            LOGGER.warning("File log write failed: %s", exc)

    def _build_snapshot(self) -> BollingerSnapshot:
        need = max(
            self.config.slow_sma,
            self.config.atr_period + 1,
            self.config.bb_period + self.config.squeeze_lookback + 2,
        )
        bars = self._fetch_bars(min_count=need)
        if len(bars) < need:
            raise RuntimeError(
                f"Not enough bars for Bollinger calculation: got={len(bars)} need={need} lookback={self.config.lookback_bars}"
            )

        last_closed = bars[-2]
        closed_bars = bars[:-1]
        closes = [bar["close"] for bar in closed_bars]
        highs = [bar["high"] for bar in closed_bars]
        lows = [bar["low"] for bar in closed_bars]

        atr = self._atr(closed_bars, self.config.atr_period)
        fast_sma = statistics.fmean(closes[-self.config.fast_sma :])
        slow_sma = statistics.fmean(closes[-self.config.slow_sma :])
        momentum = (closes[-1] - closes[-4]) / atr if atr > 0 and len(closes) >= 4 else 0.0
        momentum = self._clamp(momentum, -2.0, 2.0)

        bands = self._bollinger_bands(closes, self.config.bb_period, self.config.bb_stddev)
        history = self._bandwidth_history(closes, self.config.bb_period, self.config.bb_stddev)
        recent_history = history[-max(5, self.config.squeeze_lookback) :]
        squeeze_threshold = self._quantile(recent_history, self.config.squeeze_quantile)
        previous_bandwidth_pct = history[-2] if len(history) >= 2 else bands["bandwidth_pct"]
        bandwidth_ratio = bands["bandwidth_pct"] / max(previous_bandwidth_pct, 1e-9)
        squeeze_active = bands["bandwidth_pct"] <= squeeze_threshold
        recent_squeeze = any(
            bw <= squeeze_threshold
            for bw in history[-max(1, self.config.squeeze_release_bars + 1) : -1]
        )
        squeeze_release = bool(recent_squeeze and bandwidth_ratio >= self.config.expansion_ratio)

        score, signal, edge = self._score_bollinger_signal(
            close=closes[-1],
            atr=atr,
            fast_sma=fast_sma,
            slow_sma=slow_sma,
            momentum=momentum,
            bands=bands,
            bandwidth_ratio=bandwidth_ratio,
            squeeze_active=squeeze_active,
            squeeze_release=squeeze_release,
            squeeze_threshold=squeeze_threshold,
        )

        return BollingerSnapshot(
            bar_time=self._bar_time(last_closed),
            close=float(last_closed["close"]),
            high=float(last_closed["high"]),
            low=float(last_closed["low"]),
            atr=atr,
            fast_sma=fast_sma,
            slow_sma=slow_sma,
            momentum=momentum,
            score=score,
            signal=signal,
            bb_mid=bands["mid"],
            bb_upper=bands["upper"],
            bb_lower=bands["lower"],
            bb_position=bands["position"],
            bandwidth=bands["bandwidth"],
            bandwidth_pct=bands["bandwidth_pct"],
            bandwidth_atr=bands["bandwidth"] / max(atr, 1e-9),
            bandwidth_ratio=bandwidth_ratio,
            squeeze_threshold=squeeze_threshold,
            squeeze_active=squeeze_active,
            squeeze_release=squeeze_release,
            edge=edge,
        )

    def _bollinger_bands(self, closes: Sequence[float], period: int, stddev: float) -> Dict[str, float]:
        window = list(closes[-period:])
        mid = statistics.fmean(window)
        sigma = statistics.pstdev(window) if len(window) > 1 else 0.0
        upper = mid + sigma * stddev
        lower = mid - sigma * stddev
        bandwidth = max(upper - lower, self._point())
        position = (closes[-1] - lower) / bandwidth
        return {
            "mid": mid,
            "upper": upper,
            "lower": lower,
            "bandwidth": bandwidth,
            "bandwidth_pct": bandwidth / max(abs(mid), self._point()),
            "position": self._clamp(position, -0.5, 1.5),
        }

    def _bandwidth_history(self, closes: Sequence[float], period: int, stddev: float) -> List[float]:
        out: List[float] = []
        for end in range(period, len(closes) + 1):
            bands = self._bollinger_bands(closes[:end], period, stddev)
            out.append(bands["bandwidth_pct"])
        return out

    def _quantile(self, values: Sequence[float], quantile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        q = self._clamp(float(quantile), 0.0, 1.0)
        idx = int(round((len(ordered) - 1) * q))
        return ordered[idx]

    def _score_bollinger_signal(
        self,
        close: float,
        atr: float,
        fast_sma: float,
        slow_sma: float,
        momentum: float,
        bands: Dict[str, float],
        bandwidth_ratio: float,
        squeeze_active: bool,
        squeeze_release: bool,
        squeeze_threshold: float,
    ) -> tuple[float, str, str]:
        del squeeze_threshold
        edge = "NONE"
        direction = 0.0
        if bands["position"] >= 1.0 - self.config.bb_edge_pct:
            edge = "UPPER"
            direction = 1.0
        elif bands["position"] <= self.config.bb_edge_pct:
            edge = "LOWER"
            direction = -1.0

        if edge == "NONE" or bands["bandwidth"] / max(atr, 1e-9) < self.config.min_bandwidth_atr:
            return 0.0, "NONE", edge

        trend_alignment = 0.0
        if direction > 0 and close >= fast_sma >= slow_sma:
            trend_alignment = 0.35
        elif direction < 0 and close <= fast_sma <= slow_sma:
            trend_alignment = 0.35
        elif direction > 0 and close >= slow_sma:
            trend_alignment = 0.15
        elif direction < 0 and close <= slow_sma:
            trend_alignment = 0.15
        else:
            trend_alignment = -0.20

        momentum_alignment = self._clamp(momentum * direction, -1.0, 1.0) * 0.25
        edge_strength = abs(bands["position"] - 0.5) * 0.55
        expansion_component = self._clamp(bandwidth_ratio - 1.0, -0.4, 0.8) * 0.35
        contraction_component = 0.20 if squeeze_active else 0.0
        release_component = 0.45 if squeeze_release else 0.0
        bias_component = (self.config.long_bias - 0.5) * 0.30 * direction

        raw = edge_strength + trend_alignment + momentum_alignment + expansion_component + contraction_component + release_component + bias_component
        if not squeeze_release and not squeeze_active and bandwidth_ratio < self.config.expansion_ratio:
            raw -= 0.35

        score = self._clamp(raw * direction, -1.5, 1.5)
        signal = self._decide_signal(score)
        return score, signal, edge

    def _handle_bar(self, snapshot: MarketSnapshot) -> None:
        bb = snapshot if isinstance(snapshot, BollingerSnapshot) else None
        positions = self._positions()
        foreign_positions = self._foreign_positions()
        if bb is not None:
            self._log_file(
                "Bar %s | close=%.2f atr=%.2f bb_mid=%.2f upper=%.2f lower=%.2f pos=%.3f bw_pct=%.5f bw_ratio=%.2f squeeze=%s release=%s edge=%s score=%.2f signal=%s positions=%d foreign=%d",
                bb.bar_time,
                bb.close,
                bb.atr,
                bb.bb_mid,
                bb.bb_upper,
                bb.bb_lower,
                bb.bb_position,
                bb.bandwidth_pct,
                bb.bandwidth_ratio,
                bb.squeeze_active,
                bb.squeeze_release,
                bb.edge,
                bb.score,
                bb.signal,
                len(positions),
                len(foreign_positions),
            )
            LOGGER.info(
                "Bar %s | close=%.2f atr=%.2f edge=%s pos=%.3f bw_ratio=%.2f squeeze=%s release=%s score=%.2f signal=%s positions=%d",
                bb.bar_time,
                bb.close,
                bb.atr,
                bb.edge,
                bb.bb_position,
                bb.bandwidth_ratio,
                bb.squeeze_active,
                bb.squeeze_release,
                bb.score,
                bb.signal,
                len(positions),
            )
        else:
            LOGGER.info("Bar %s | close=%.2f signal=%s positions=%d", snapshot.bar_time, snapshot.close, snapshot.signal, len(positions))

        if not positions:
            if snapshot.signal != "NONE" and self._cooldown_ok(snapshot.bar_time):
                self._log_file("No open position; Bollinger signal=%s", snapshot.signal)
                self._enter(snapshot)
            return

        pos = positions[0]
        if self._should_roll(snapshot, pos):
            self._log_file("Roll trigger met, closing and re-evaluating on next bar.")
            LOGGER.info("Roll trigger met, closing and re-evaluating on next bar.")
            self.close_all_positions()
            self._last_signal = "NONE"
            self._last_trade_close_time = snapshot.bar_time
            self._pending_reverse_signal = "NONE"
            self._pending_reverse_since = None
            return

        if self._should_reverse(snapshot, pos):
            self._handle_reverse_signal(snapshot, pos)
            return

        self._pending_reverse_signal = "NONE"
        self._pending_reverse_since = None
        self._maybe_trail(snapshot, pos)

    def _handle_reverse_signal(self, snapshot: MarketSnapshot, pos: PositionState) -> None:
        del pos
        now = snapshot.bar_time
        if self._pending_reverse_signal != snapshot.signal:
            self._pending_reverse_signal = snapshot.signal
            self._pending_reverse_since = now
            self._log_file("Reverse Bollinger signal detected; waiting 15 minutes before reversing.")
            LOGGER.info("Reverse Bollinger signal detected; waiting 15 minutes before reversing.")
            return

        if self._pending_reverse_since is not None:
            elapsed_minutes = (now - self._pending_reverse_since).total_seconds() / 60.0
            if elapsed_minutes < 15.0:
                self._log_file("Reverse signal pending for %.1f minutes; need 15.0 minutes before reversing.", elapsed_minutes)
                LOGGER.info("Reverse signal pending for %.1f minutes; need 15.0 minutes before reversing.", elapsed_minutes)
                return

        self._log_file("Reverse signal persisted 15 minutes; closing and entering new direction.")
        LOGGER.info("Reverse signal persisted 15 minutes; closing and entering new direction.")
        self.close_all_positions()
        self._last_trade_close_time = snapshot.bar_time
        self._pending_reverse_signal = "NONE"
        self._pending_reverse_since = None
        if self._cooldown_ok(snapshot.bar_time):
            self._enter(snapshot)

    def _order_request(self, direction: str, volume: float, price: float, sl: float, tp: float) -> Dict[str, Any]:
        request = super()._order_request(direction, volume, price, sl, tp)
        request["comment"] = "bb-edge-demo" if not self.config.live else "bb-edge-live"
        return request


def build_parser() -> argparse.ArgumentParser:
    parser = build_doomsday_parser()
    parser.description = "MT5 Bollinger edge/squeeze strategy with doomsday-grade controls."
    parser.add_argument("--bb-period", type=int, default=20, help="Bollinger lookback period.")
    parser.add_argument("--bb-stddev", type=float, default=2.0, help="Bollinger standard-deviation multiplier.")
    parser.add_argument("--bb-edge-pct", type=float, default=0.16, help="Band-position edge threshold, 0.16 means top/bottom 16%%.")
    parser.add_argument("--squeeze-lookback", type=int, default=80, help="Bars used to rank Bollinger bandwidth.")
    parser.add_argument("--squeeze-quantile", type=float, default=0.25, help="Bandwidth quantile considered a squeeze.")
    parser.add_argument("--expansion-ratio", type=float, default=1.12, help="Current/previous bandwidth ratio for expansion.")
    parser.add_argument("--min-bandwidth-atr", type=float, default=0.35, help="Minimum band width in ATR multiples.")
    parser.add_argument("--squeeze-release-bars", type=int, default=6, help="Recent bars checked for squeeze release.")
    parser.add_argument("--log-file", default=LOG_FILE_DEFAULT, help="Strategy file log path.")
    parser.add_argument("--state-path", default=None, help="Accepted for supervisor/auto-quant compatibility.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    terminal_path = args.terminal_path.strip() or None
    config = BollingerStrategyConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        host=args.host,
        port=args.port,
        live=bool(args.live),
        risk_pct=float(args.risk_pct),
        stop_atr=float(args.stop_atr),
        reward_multiple=float(args.reward_multiple),
        tp_min_usd=float(args.tp_min_usd),
        tp_max_usd=float(args.tp_max_usd),
        long_bias=float(args.long_bias),
        trend_threshold=float(args.trend_threshold),
        roll_trigger_pct=float(args.roll_trigger_pct),
        cooldown_minutes=int(args.cooldown_minutes),
        max_leverage=float(args.max_leverage),
        max_drawdown_pct=float(args.max_drawdown_pct),
        max_daily_loss_pct=float(args.max_daily_loss_pct),
        max_lots=float(args.max_lots),
        magic=int(args.magic),
        deviation=int(args.deviation),
        loop_seconds=int(args.loop_seconds),
        lookback_bars=int(args.lookback_bars),
        atr_period=int(args.atr_period),
        fast_sma=int(args.fast_sma),
        slow_sma=int(args.slow_sma),
        log_level=args.log_level,
        terminal_path=terminal_path,
        bb_period=int(args.bb_period),
        bb_stddev=float(args.bb_stddev),
        bb_edge_pct=float(args.bb_edge_pct),
        squeeze_lookback=int(args.squeeze_lookback),
        squeeze_quantile=float(args.squeeze_quantile),
        expansion_ratio=float(args.expansion_ratio),
        min_bandwidth_atr=float(args.min_bandwidth_atr),
        squeeze_release_bars=int(args.squeeze_release_bars),
        log_file=str(args.log_file),
        state_path=str(args.state_path) if args.state_path else None,
    )

    strategy = BollingerEdgeSqueezeMT5Strategy(config)
    strategy.run()


if __name__ == "__main__":
    main()
