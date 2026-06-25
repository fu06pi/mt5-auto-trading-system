import datetime as dt
import importlib.util
import sys
import types
from pathlib import Path


class FakeMT5:
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    TRADE_ACTION_DEAL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC = 1
    TRADE_RETCODE_DONE = 10009
    TIMEFRAME_M1 = 1
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_M30 = 30
    TIMEFRAME_H1 = 60

    def __init__(self, *args, **kwargs):
        self._equity = 100000.0
        self.requests = []

    def initialize(self, *args, **kwargs):
        return True

    def account_info(self):
        return types.SimpleNamespace(equity=self._equity, balance=self._equity)

    def symbol_info(self, symbol):
        return types.SimpleNamespace(volume_min=0.01, volume_max=100.0, volume_step=0.01)

    def symbol_info_tick(self, symbol):
        return types.SimpleNamespace(bid=4399.5, ask=4400.5)

    def order_send(self, request):
        self.requests.append(request)
        return types.SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, order=12345)

    def positions_get(self, symbol=None):
        return []

    def last_error(self):
        return None


def load_strategy_module():
    fake_module = types.ModuleType("pymt5linux")
    fake_module.MetaTrader5 = FakeMT5
    sys.modules["pymt5linux"] = fake_module
    path = Path("/home/chain4655/Documents/Projects/MT5/strategies/mt5_xauusd_asian_reversal_strategy.py")
    spec = importlib.util.spec_from_file_location("mt5_xauusd_asian_reversal_strategy_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_config(module, tmp_path, **overrides):
    data = dict(
        symbol="XAUUSD",
        timeframe="M1",
        host="127.0.0.1",
        port=18812,
        live=True,
        start_equity=100000.0,
        daily_dd_limit=0.03,
        total_dd_limit=0.10,
        profit_target=0.05,
        risk_pct=0.0035,
        max_lots=0.5,
        atr_period=14,
        stop_atr=1.5,
        take_profit_atr=6.0,
        pivot_lookback=4,
        stall_bars=3,
        stall_range_atr=0.5,
        stall_vol_ratio=0.9,
        reversal_atr=0.25,
        session_start_hour=0,
        session_end_hour=7,
        max_trades_per_day=1,
        max_hold_minutes=120,
        max_consecutive_losses=3,
        loss_cooldown_minutes=120,
        max_concentration_share=0.55,
        min_positive_days_for_concentration=999999,
        cooldown_bars_after_trade=2,
        startup_warmup_bars=1,
        loop_seconds=5,
        lookback_bars=120,
        max_spread_points=120.0,
        state_path=str(tmp_path / "asrv_state.json"),
        log_file=str(tmp_path / "asrv.log"),
        deviation=30,
        magic=204574,
        log_level="INFO",
        max_leverage=5.0,
        order_comment="asrv-204574",
    )
    data.update(overrides)
    return module.StrategyConfig(**data)


def test_asian_reversal_entry_request_uses_trade_action_deal_and_type(tmp_path):
    module = load_strategy_module()
    strategy = module.XAUUSDAsianReversalStrategy(make_config(module, tmp_path))
    strategy._ensure_mt5 = lambda: None
    strategy._save_state = lambda: None

    snapshot = module.MarketSnapshot(
        bar_time=dt.datetime(2026, 6, 18, 2, 0),
        close=4400.0,
        high=4401.0,
        low=4399.0,
        atr=2.0,
        session="asia",
        spread_points=40.0,
    )

    strategy._enter("BUY", snapshot)

    request = strategy.mt5.requests[-1]
    assert request["action"] == FakeMT5.TRADE_ACTION_DEAL
    assert request["type"] == FakeMT5.ORDER_TYPE_BUY
    assert request["comment"] == "asrv-204574"


def test_asian_reversal_close_request_uses_trade_action_deal_and_opposite_type(tmp_path):
    module = load_strategy_module()
    strategy = module.XAUUSDAsianReversalStrategy(make_config(module, tmp_path))
    strategy._ensure_mt5 = lambda: None
    strategy._positions = lambda: [
        module.PositionState(
            ticket=9,
            symbol="XAUUSD",
            magic=204574,
            type=FakeMT5.POSITION_TYPE_BUY,
            volume=0.2,
            price_open=4400.0,
            sl=4390.0,
            tp=4420.0,
            profit=10.0,
        )
    ]

    assert strategy.close_all_positions()

    request = strategy.mt5.requests[-1]
    assert request["action"] == FakeMT5.TRADE_ACTION_DEAL
    assert request["type"] == FakeMT5.ORDER_TYPE_SELL
    assert request["position"] == 9
