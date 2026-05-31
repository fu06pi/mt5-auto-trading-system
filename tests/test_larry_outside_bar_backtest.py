import datetime as dt
import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(
    "/home/chain4655/Documents/backtest_reports/larry_outside_bar_xauusd/"
    "larry_outside_bar_backtest.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("larry_outside_bar_backtest", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_bar(day, open_price, high, low, close):
    module = load_module()
    return module.Bar(
        time=dt.datetime(2026, 1, day),
        open=float(open_price),
        high=float(high),
        low=float(low),
        close=float(close),
        volume=1000.0,
    )


def test_bearish_outside_close_below_previous_low_is_next_day_buy_signal():
    module = load_module()
    previous = make_bar(1, 100, 105, 95, 101)
    current = make_bar(2, 101, 108, 93, 94)

    signal = module.outside_bar_signal(previous, current, min_body_ratio=1.5)

    assert signal == "BUY"


def test_bullish_outside_close_above_previous_high_is_next_day_sell_signal():
    module = load_module()
    previous = make_bar(1, 100, 105, 95, 99)
    current = make_bar(2, 99, 108, 92, 106)

    signal = module.outside_bar_signal(previous, current, min_body_ratio=1.5)

    assert signal == "SELL"


def test_small_body_outside_bar_is_filtered_out():
    module = load_module()
    previous = make_bar(1, 100, 105, 95, 101)
    current = make_bar(2, 100, 108, 92, 94)

    signal = module.outside_bar_signal(previous, current, min_body_ratio=10.0)

    assert signal == "NONE"


def test_fpo_exits_buy_on_next_profitable_open():
    module = load_module()
    bars = [
        make_bar(1, 100, 105, 95, 101),
        make_bar(2, 101, 108, 93, 94),
        make_bar(3, 96, 98, 95.5, 97),
        make_bar(4, 99, 101, 98, 100),
    ]

    result = module.run_backtest(
        bars=bars,
        initial_equity=10000.0,
        notional=1000.0,
        min_body_ratio=1.0,
        atr_period=2,
        stop_buffer_atr=0.0,
        max_hold_days=5,
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.direction == "BUY"
    assert trade.entry_time == bars[2].time
    assert trade.exit_time == bars[3].time
    assert trade.exit_reason == "FPO_PROFIT_OPEN"
    assert trade.pnl > 0
