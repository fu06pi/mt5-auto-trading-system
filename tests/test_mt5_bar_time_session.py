import datetime as dt

from test_xauusd_trend_strategy_risk_controls import load_strategy_module, make_config


def test_bar_time_uses_mt5_chart_timestamp_without_local_timezone_shift(tmp_path):
    module = load_strategy_module()
    config = make_config(module, tmp_path)
    strategy = module.XAUUSDTrendStrategy(config)

    bar = {"time": 1780505700.0}

    assert strategy._bar_time(bar) == dt.datetime(2026, 6, 3, 16, 55, 0)
    assert strategy._session_label(strategy._bar_time(bar)) == "us_london_overlap"


def test_max_hold_uses_same_mt5_chart_time_basis_for_position_open_time(tmp_path):
    module = load_strategy_module()
    config = make_config(module, tmp_path, max_hold_minutes=30)
    strategy = module.XAUUSDTrendStrategy(config)
    pos = module.PositionState(
        ticket=1,
        symbol="XAUUSD",
        magic=26052031,
        type=module.MetaTrader5.POSITION_TYPE_SELL,
        volume=0.1,
        price_open=4450.0,
        sl=4460.0,
        tp=4430.0,
        profit=0.0,
        time_open=1780505700,
    )

    assert not strategy._max_hold_exceeded(pos, dt.datetime(2026, 6, 3, 17, 20, 0))
    assert strategy._max_hold_exceeded(pos, dt.datetime(2026, 6, 3, 17, 25, 0))
