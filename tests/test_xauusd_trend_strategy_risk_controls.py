import datetime as dt
import importlib.util
import sys
import types
from pathlib import Path


class FakeMT5:
    default_positions = []
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_SLTP = 2
    ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_BOC = 3
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_PLACED = 10008
    TRADE_RETCODE_DONE_PARTIAL = 10010
    TRADE_RETCODE_INVALID_FILL = 10030
    TRADE_RETCODE_INVALID_ORDER = 10035
    TIMEFRAME_M1 = 1
    TIMEFRAME_M2 = 2
    TIMEFRAME_M3 = 3
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_M30 = 30
    TIMEFRAME_H1 = 60
    TIMEFRAME_H4 = 240
    TIMEFRAME_D1 = 1440

    def __init__(self, *args, **kwargs):
        self._equity = 100000.0
        self._positions = list(self.default_positions)

    def initialize(self, *args, **kwargs):
        return True

    def shutdown(self):
        return True

    def account_info(self):
        return types.SimpleNamespace(equity=self._equity, balance=self._equity, margin_free=self._equity)

    def symbol_info(self, symbol):
        return types.SimpleNamespace(
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            trade_contract_size=100.0,
            point=0.01,
            digits=2,
            filling_mode=1,
        )

    def order_calc_profit(self, order_type, symbol, volume, price, sl):
        return abs(price - sl) * 100.0 * volume

    def order_calc_margin(self, order_type, symbol, volume, price):
        return 1000.0 * volume

    def positions_get(self, symbol=None):
        return self._positions

    def last_error(self):
        return None


def load_strategy_module():
    fake_module = types.ModuleType("pymt5linux")
    fake_module.MetaTrader5 = FakeMT5
    sys.modules["pymt5linux"] = fake_module
    path = Path("/home/chain4655/Documents/Projects/MT5/strategies/mt5_xauusd_trend_strategy.py")
    spec = importlib.util.spec_from_file_location("mt5_xauusd_trend_strategy_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_config(module, tmp_path, **overrides):
    data = dict(
        symbol="XAUUSD",
        timeframe="M5",
        host="127.0.0.1",
        port=18812,
        live=False,
        start_equity=100000.0,
        daily_dd_limit=0.03,
        total_dd_limit=0.045,
        profit_target=0.10,
        risk_pct=0.01,
        max_lots=10.0,
        max_lots_per_order=10.0,
        max_leverage=5.0,
        fast_sma=20,
        slow_sma=60,
        htf_fast_sma=50,
        htf_slow_sma=200,
        trend_threshold=0.35,
        htf_comp_momentum_threshold=1.10,
        htf_momentum_bias_weight=0.0,
        momentum_score_weight=0.25,
        atr_period=14,
        breakout_lookback=20,
        enable_false_breakout_reversal=False,
        false_breakout_direction="SELL_ONLY",
        false_breakout_lookback=20,
        false_breakout_min_atr=0.15,
        false_breakout_close_back_atr=0.05,
        false_breakout_wick_ratio=0.45,
        stop_atr=2.5,
        reward_multiple=2.5,
        trail_trigger_atr=1.5,
        trail_lock_atr=0.5,
        trail_stable_minutes=0,
        trail_same_direction_cooldown_minutes=60,
        break_even_atr=1.0,
        break_even_lock_atr=0.15,
        fee_cover_price_offset=0.1,
        session_start_utc=0,
        session_end_utc=0,
        max_spread_points=120.0,
        max_trades_per_day=999,
        max_consecutive_losses=9999,
        loss_cooldown_losses=3,
        loss_cooldown_minutes=60,
        auto_half_profit_usd=2000.0,
        auto_half_fraction=0.5,
        cooldown_bars_after_trade=2,
        startup_warmup_bars=0,
        max_hold_minutes=180,
        loop_seconds=10,
        lookback_bars=260,
        htf_lookback_bars=400,
        max_concentration_share=0.45,
        min_positive_days_for_concentration=3,
        allow_pyramiding=False,
        allow_foreign_positions=False,
        state_path=str(tmp_path / "state.json"),
        log_file=str(tmp_path / "strategy.log"),
        terminal_path=None,
        deviation=30,
        magic=26052031,
        log_level="INFO",
        warmup_risk_days=7,
        warmup_risk_multiplier=0.5,
        half_close_cooldown_bars=24,
        primary_tp_reward_multiple=0.0,
    )
    data.update(overrides)
    return module.StrategyConfig(**data)


def test_first_week_position_size_uses_half_risk(tmp_path):
    module = load_strategy_module()
    config = make_config(module, tmp_path, risk_pct=0.01, warmup_risk_days=7, warmup_risk_multiplier=0.5)
    strategy = module.XAUUSDTrendStrategy(config)
    strategy.state.initial_equity = 100000.0
    strategy.state.current_day = "2026-05-27"
    strategy.mt5._equity = 100000.0

    volume = strategy._size_position("BUY", price=3300.0, sl=3290.0)

    assert volume == 0.5


def test_warmup_risk_expires_after_configured_days(tmp_path):
    module = load_strategy_module()
    config = make_config(module, tmp_path, risk_pct=0.01, warmup_risk_days=7, warmup_risk_multiplier=0.5)
    strategy = module.XAUUSDTrendStrategy(config)
    strategy.state.initial_equity = 100000.0
    strategy.state.current_day = "2026-06-04"
    strategy.state.risk_warmup_started_at = "2026-05-27T00:00:00"
    strategy.mt5._equity = 100000.0

    volume = strategy._size_position("BUY", price=3300.0, sl=3290.0)

    assert volume == 1.0


def test_half_close_starts_reentry_cooldown(tmp_path):
    module = load_strategy_module()
    config = make_config(module, tmp_path, half_close_cooldown_bars=24, loop_seconds=300)
    strategy = module.XAUUSDTrendStrategy(config)
    strategy.state.auto_half_close_done = True
    strategy.state.last_half_close_bar_time = "2026-05-27T10:00:00"

    assert not strategy._cooldown_ok(dt.datetime(2026, 5, 27, 10, 30, 0))
    assert strategy._cooldown_ok(dt.datetime(2026, 5, 27, 12, 5, 0))


def test_build_sl_tp_allows_profit_target_to_be_pulled_closer_than_campaign_rr(tmp_path):
    module = load_strategy_module()
    config = make_config(module, tmp_path, stop_atr=2.5, reward_multiple=2.5, primary_tp_reward_multiple=1.2)
    strategy = module.XAUUSDTrendStrategy(config)
    strategy._symbol_point = 0.01
    strategy._symbol_digits = 2

    sl, tp = strategy._build_sl_tp("BUY", price=3300.0, atr=4.0)

    assert sl == 3290.0
    assert tp == 3312.0


def test_positions_are_read_through_fresh_client_when_main_client_is_stale(tmp_path):
    module = load_strategy_module()
    FakeMT5.default_positions = [
        types.SimpleNamespace(
            ticket=1,
            symbol="XAUUSD",
            magic=26052031,
            type=FakeMT5.POSITION_TYPE_SELL,
            volume=0.3,
            price_open=4450.0,
            sl=4470.0,
            tp=4420.0,
            profit=100.0,
            time=0,
        )
    ]
    config = make_config(module, tmp_path)
    strategy = module.XAUUSDTrendStrategy(config)
    strategy.mt5._positions = []

    positions = strategy._positions()

    assert len(positions) == 1
    assert positions[0].ticket == 1
    FakeMT5.default_positions = []
