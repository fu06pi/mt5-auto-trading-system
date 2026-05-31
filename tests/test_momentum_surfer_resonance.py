import datetime as dt
import importlib.util
import sys
import types
from pathlib import Path


class FakeMetaTrader5:
    TIMEFRAME_M1 = 1
    TIMEFRAME_M2 = 2
    TIMEFRAME_M3 = 3
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_M30 = 30
    TIMEFRAME_H1 = 60
    TIMEFRAME_H4 = 240
    TIMEFRAME_D1 = 1440

    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1

    TRADE_ACTION_DEAL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2
    ORDER_FILLING_BOC = 3

    def __init__(self, *args, **kwargs):
        pass


def load_strategy_module():
    fake_module = types.ModuleType("pymt5linux")
    fake_module.MetaTrader5 = FakeMetaTrader5
    sys.modules.setdefault("pymt5linux", fake_module)

    path = Path("/home/chain4655/Documents/Projects/MT5/strategies/mt5_xauusd_momentum_surfer_strategy.py")
    spec = importlib.util.spec_from_file_location("momentum_surfer_strategy", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def bar(index, open_price, high, low, close, volume=100):
    base = dt.datetime(2026, 5, 18, 8, 0)
    return {
        "time": (base + dt.timedelta(minutes=15 * index)).timestamp(),
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "tick_volume": volume,
    }


def trend_bars(direction="up"):
    bars = []
    price = 3300.0
    for idx in range(70):
        drift = 0.55 if direction == "up" else -0.55
        open_price = price
        close = price + drift
        bars.append(bar(idx, open_price, max(open_price, close) + 0.7, min(open_price, close) - 0.7, close))
        price = close
    return bars


def test_resonance_signal_buys_after_ema_avwap_support_sweep_and_reclaim():
    module = load_strategy_module()
    strategy = module.XAUUSDMomentumSurferStrategy.__new__(module.XAUUSDMomentumSurferStrategy)

    bars = trend_bars("up")
    # Pullback into EMA9/EMA21/session-AVWAP confluence, then fast bullish reclaim.
    bars[-3] = bar(67, 3336.0, 3337.0, 3331.5, 3332.5, 130)
    bars[-2] = bar(68, 3332.4, 3334.2, 3328.0, 3333.7, 180)
    bars[-1] = bar(69, 3333.8, 3339.0, 3332.9, 3338.5, 220)

    signal, strength, meta = strategy._decide_resonance_signal(
        bars,
        atr=4.0,
        spread_points=50.0,
        max_spread_points=150.0,
        compression_atr=2.5,
        sweep_lookback=12,
        reclaim_body_min_atr=0.25,
    )

    assert signal == "BUY"
    assert strength > 0.0
    assert meta["trend"] == "UP"
    assert meta["swept_zone"] is True
    assert meta["reclaimed"] is True


def test_resonance_signal_blocks_without_liquidity_sweep():
    module = load_strategy_module()
    strategy = module.XAUUSDMomentumSurferStrategy.__new__(module.XAUUSDMomentumSurferStrategy)

    bars = trend_bars("up")
    bars[-2] = bar(68, 3335.0, 3337.0, 3334.1, 3336.5, 180)
    bars[-1] = bar(69, 3336.6, 3339.2, 3336.0, 3338.8, 220)

    signal, strength, meta = strategy._decide_resonance_signal(
        bars,
        atr=4.0,
        spread_points=50.0,
        max_spread_points=150.0,
        compression_atr=2.5,
        sweep_lookback=12,
        reclaim_body_min_atr=0.25,
    )

    assert signal == "NONE"
    assert strength == 0.0
    assert meta["swept_zone"] is False


def test_resonance_signal_sells_after_resistance_sweep_and_rejection():
    module = load_strategy_module()
    strategy = module.XAUUSDMomentumSurferStrategy.__new__(module.XAUUSDMomentumSurferStrategy)

    bars = trend_bars("down")
    bars[-3] = bar(67, 3264.0, 3268.0, 3262.8, 3267.2, 130)
    bars[-2] = bar(68, 3267.1, 3271.6, 3265.5, 3266.0, 180)
    bars[-1] = bar(69, 3265.9, 3266.5, 3260.8, 3261.1, 220)

    signal, strength, meta = strategy._decide_resonance_signal(
        bars,
        atr=4.0,
        spread_points=50.0,
        max_spread_points=150.0,
        compression_atr=2.5,
        sweep_lookback=12,
        reclaim_body_min_atr=0.25,
    )

    assert signal == "SELL"
    assert strength > 0.0
    assert meta["trend"] == "DOWN"
    assert meta["swept_zone"] is True
    assert meta["reclaimed"] is True
