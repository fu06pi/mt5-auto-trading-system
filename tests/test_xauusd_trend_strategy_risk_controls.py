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
    DEAL_ENTRY_IN = 0
    DEAL_ENTRY_OUT = 1
    DEAL_ENTRY_INOUT = 2
    DEAL_ENTRY_OUT_BY = 3
    DEAL_TYPE_BUY = 0
    DEAL_TYPE_SELL = 1

    def __init__(self, *args, **kwargs):
        self._equity = 100000.0
        self._positions = list(self.default_positions)
        self._deals = []

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

    def history_deals_get(self, from_time, to_time):
        return self._deals

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
        htf_timeframe="H1",
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
        profit_close_usd=0.0,
        profit_close_pause_minutes=0,
        loss_close_pause_minutes=0,
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


def test_friday_force_close_closes_owned_positions_and_blocks_reentry(tmp_path):
    module = load_strategy_module()
    config = make_config(
        module,
        tmp_path,
        force_close_friday_hour_utc=12,
        force_close_friday_minute_utc=0,
    )
    strategy = module.XAUUSDTrendStrategy(config)
    closed = []
    strategy._positions = lambda: [
        module.PositionState(
            ticket=1,
            symbol="XAUUSD",
            magic=26052031,
            type=FakeMT5.POSITION_TYPE_BUY,
            volume=0.5,
            price_open=4400.0,
            sl=4380.0,
            tp=4450.0,
            profit=25.0,
            time_open=None,
        )
    ]
    strategy.close_all_positions = lambda: closed.append(True) or True

    assert not strategy._friday_force_close_cutoff_reached(dt.datetime(2026, 6, 19, 11, 59, 0))
    assert strategy._maybe_force_close_friday(dt.datetime(2026, 6, 19, 12, 0, 0))
    assert closed == [True]
    assert strategy._friday_force_close_cutoff_reached(dt.datetime(2026, 6, 19, 12, 1, 0))


def test_friday_force_close_disabled_by_default(tmp_path):
    module = load_strategy_module()
    config = make_config(module, tmp_path)
    strategy = module.XAUUSDTrendStrategy(config)

    assert not strategy._friday_force_close_cutoff_reached(dt.datetime(2026, 6, 19, 12, 0, 0))
    assert not strategy._maybe_force_close_friday(dt.datetime(2026, 6, 19, 12, 0, 0))


def test_chop_risk_multiplier_scales_per_order_cap(tmp_path):
    module = load_strategy_module()
    config = make_config(
        module,
        tmp_path,
        risk_pct=0.10,
        max_lots=10.0,
        max_lots_per_order=1.0,
        warmup_risk_days=0,
    )
    strategy = module.XAUUSDTrendStrategy(config)
    strategy.mt5._equity = 100000.0

    volume = strategy._size_position("BUY", price=3300.0, sl=3290.0, risk_multiplier=0.25)

    assert volume == 0.25


def test_chop_risk_multiplier_below_one_blocks_pyramiding_entries(tmp_path):
    module = load_strategy_module()
    config = make_config(module, tmp_path, allow_pyramiding=True)
    strategy = module.XAUUSDTrendStrategy(config)
    strategy._positions = lambda: [
        module.PositionState(
            ticket=1,
            symbol="XAUUSD",
            magic=26052031,
            type=FakeMT5.POSITION_TYPE_SELL,
            volume=0.5,
            price_open=4450.0,
            sl=4470.0,
            tp=4420.0,
            profit=50.0,
            time_open=None,
        )
    ]
    strategy._foreign_positions = lambda: []
    strategy._sync_closed_trades = lambda: None
    strategy._maybe_auto_half_close = lambda positions: None
    strategy._record_startup_warmup_bar = lambda snapshot: False
    strategy._session_allowed = lambda bar_time: True
    strategy._loss_cooldown_active = lambda: False
    strategy._cooldown_ok = lambda bar_time: True
    strategy._maybe_trail = lambda snapshot, position: None
    entered = []
    strategy._enter = lambda snapshot: entered.append(snapshot.signal)
    snapshot = module.MarketSnapshot(
        bar_time=dt.datetime(2026, 6, 2, 10, 0, 0),
        close=4450.0,
        high=4455.0,
        low=4445.0,
        atr=5.0,
        fast_sma=4451.0,
        slow_sma=4452.0,
        htf_fast_sma=4460.0,
        htf_slow_sma=4470.0,
        htf_signal="BEAR",
        compensated_htf_signal="BEAR",
        spread_points=40.0,
        momentum=-0.5,
        m15_momentum=-0.2,
        score=-0.6,
        signal="SELL",
        signal_source="trend",
        session="london_pre_us",
        chop_is_chop=True,
        chop_points=4,
        chop_reason="efficiency+flat_sma+weak_score+alternating_signals",
        chop_risk_multiplier=0.25,
        false_breakout_signal="NONE",
        false_breakout_reason="none",
    )

    strategy._handle_bar(snapshot)

    assert entered == []


def test_htf_lag_reversal_guard_blocks_sell_when_bear_htf_lags_bullish_m5(tmp_path):
    module = load_strategy_module()
    config = make_config(
        module,
        tmp_path,
        enable_htf_lag_reversal_guard=True,
        htf_lag_momentum_threshold=0.70,
        htf_lag_m15_threshold=0.50,
        htf_lag_close_sma_buffer_atr=0.05,
    )
    strategy = module.XAUUSDTrendStrategy(config)

    blocked = strategy._htf_lag_reversal_blocks(
        signal="SELL",
        htf_signal="BEAR",
        close=4460.0,
        fast_sma=4458.0,
        slow_sma=4452.0,
        momentum=0.80,
        m15_momentum=0.60,
        atr=6.0,
    )

    assert blocked


def test_htf_lag_reversal_guard_does_not_block_aligned_sell(tmp_path):
    module = load_strategy_module()
    config = make_config(
        module,
        tmp_path,
        enable_htf_lag_reversal_guard=True,
        htf_lag_momentum_threshold=0.70,
        htf_lag_m15_threshold=0.50,
        htf_lag_close_sma_buffer_atr=0.05,
    )
    strategy = module.XAUUSDTrendStrategy(config)

    blocked = strategy._htf_lag_reversal_blocks(
        signal="SELL",
        htf_signal="BEAR",
        close=4449.0,
        fast_sma=4458.0,
        slow_sma=4452.0,
        momentum=-0.40,
        m15_momentum=-0.30,
        atr=6.0,
    )

    assert not blocked

def test_htf_filter_uses_configured_timeframe(tmp_path):
    module = load_strategy_module()
    config = make_config(module, tmp_path, htf_timeframe="H4", htf_fast_sma=3, htf_slow_sma=5)
    strategy = module.XAUUSDTrendStrategy(config)
    captured = []
    bars = [
        {"time": i, "open": 100.0 + i, "high": 101.0 + i, "low": 99.0 + i, "close": 100.0 + i, "tick_volume": 1}
        for i in range(8)
    ]

    def fake_copy_rates_once(timeframe, bars_count):
        captured.append((timeframe, bars_count))
        return bars, None

    strategy._copy_rates_once = fake_copy_rates_once

    _, _, signal = strategy._build_htf_filter()

    assert captured == [(FakeMT5.TIMEFRAME_H4, 400)]
    assert signal == "BULL"



def test_loss_close_pause_blocks_only_until_expiry(tmp_path):
    module = load_strategy_module()
    config = make_config(module, tmp_path, loss_close_pause_minutes=60)
    strategy = module.XAUUSDTrendStrategy(config)

    strategy._activate_loss_close_pause(-125.0)

    assert strategy.state.loss_pause_until is not None
    assert strategy._loss_close_pause_active(dt.datetime.now() + dt.timedelta(minutes=30))
    assert not strategy._loss_close_pause_active(dt.datetime.now() + dt.timedelta(minutes=61))
    assert strategy.state.loss_pause_until is None


def test_profit_pause_can_be_disabled_without_blocking_loss_close_pause(tmp_path):
    module = load_strategy_module()
    config = make_config(
        module,
        tmp_path,
        profit_close_pause_minutes=0,
        loss_close_pause_minutes=60,
    )
    strategy = module.XAUUSDTrendStrategy(config)

    strategy._activate_profit_pause(500.0)
    strategy._activate_loss_close_pause(-50.0)

    assert strategy.state.profit_pause_until is None
    assert strategy.state.loss_pause_until is not None


def test_pullback_entry_gate_blocks_chasing_far_from_fast_sma(tmp_path):
    module = load_strategy_module()
    config = make_config(module, tmp_path, entry_mode="pullback", pullback_max_atr=0.35)
    strategy = module.XAUUSDTrendStrategy(config)
    bars = [{"open": 100.0, "high": 112.0, "low": 109.0, "close": 111.0}]

    allowed = strategy._trend_pullback_allows("BUY", bars, atr=10.0, fast_sma=100.0)

    assert not allowed


def test_pullback_entry_gate_allows_retest_and_close_back_with_trend(tmp_path):
    module = load_strategy_module()
    config = make_config(module, tmp_path, entry_mode="pullback", pullback_max_atr=0.35)
    strategy = module.XAUUSDTrendStrategy(config)
    buy_bars = [{"open": 101.0, "high": 106.0, "low": 103.0, "close": 104.5}]
    sell_bars = [{"open": 99.0, "high": 97.0, "low": 93.0, "close": 95.5}]

    assert strategy._trend_pullback_allows("BUY", buy_bars, atr=10.0, fast_sma=100.0)
    assert strategy._trend_pullback_allows("SELL", sell_bars, atr=10.0, fast_sma=100.0)


def test_closed_trade_sync_counts_only_owned_real_closing_deals(tmp_path):
    module = load_strategy_module()
    config = make_config(
        module,
        tmp_path,
        symbol="US100.cash",
        magic=909001,
        loss_cooldown_losses=3,
        loss_cooldown_minutes=120,
    )
    strategy = module.XAUUSDTrendStrategy(config)
    base = int(dt.datetime(2026, 6, 12, 22, 0, tzinfo=dt.UTC).timestamp())
    strategy.mt5._deals = [
        types.SimpleNamespace(ticket=1, time=base + 10, symbol="US100.cash", magic=909001, entry=FakeMT5.DEAL_ENTRY_OUT, profit=-100.0, commission=0.0, swap=0.0),
        types.SimpleNamespace(ticket=2, time=base + 20, symbol="XAUUSD", magic=909001, entry=FakeMT5.DEAL_ENTRY_OUT, profit=500.0, commission=0.0, swap=0.0),
        types.SimpleNamespace(ticket=3, time=base + 30, symbol="US100.cash", magic=0, entry=FakeMT5.DEAL_ENTRY_OUT, profit=500.0, commission=0.0, swap=0.0),
        types.SimpleNamespace(ticket=4, time=base + 40, symbol="US100.cash", magic=909001, entry=FakeMT5.DEAL_ENTRY_OUT, profit=2.5, commission=0.0, swap=0.0),
        types.SimpleNamespace(ticket=5, time=base + 50, symbol="US100.cash", magic=909001, entry=FakeMT5.DEAL_ENTRY_OUT, profit=-150.0, commission=0.0, swap=0.0),
        types.SimpleNamespace(ticket=6, time=base + 60, symbol="US100.cash", magic=909001, entry=FakeMT5.DEAL_ENTRY_IN, profit=-999.0, commission=0.0, swap=0.0),
        types.SimpleNamespace(ticket=7, time=base + 70, symbol="US100.cash", magic=909001, entry=FakeMT5.DEAL_ENTRY_OUT, profit=-200.0, commission=0.0, swap=0.0),
    ]

    strategy._sync_closed_trades()

    assert strategy.state.consecutive_losses == 3
    assert strategy.state.loss_cooldown_until is not None
    assert strategy.state.last_close_profit == -200.0


def test_enter_hard_blocks_before_order_send_when_loss_cooldown_active(tmp_path):
    module = load_strategy_module()
    config = make_config(module, tmp_path, live=True)
    strategy = module.XAUUSDTrendStrategy(config)
    strategy.state.consecutive_losses = 3
    strategy.state.loss_cooldown_until = (dt.datetime.now() + dt.timedelta(minutes=30)).isoformat(timespec="seconds")
    snapshot = module.MarketSnapshot(
        bar_time=dt.datetime(2026, 6, 12, 23, 0),
        close=3300.0,
        high=3305.0,
        low=3295.0,
        atr=10.0,
        fast_sma=3290.0,
        slow_sma=3280.0,
        htf_fast_sma=3300.0,
        htf_slow_sma=3290.0,
        htf_signal="BULL",
        compensated_htf_signal="BULL",
        spread_points=10.0,
        momentum=1.0,
        m15_momentum=1.0,
        score=1.0,
        signal="BUY",
        signal_source="primary",
        session="ny",
        chop_is_chop=False,
        chop_points=0,
        chop_reason="none",
        chop_risk_multiplier=1.0,
        false_breakout_signal="NONE",
        false_breakout_reason="none",
    )
    called = []
    strategy._tick_prices = lambda: called.append("tick") or (3301.0, 3300.0)
    strategy._send_order_with_filling_fallback = lambda request: called.append("send")

    strategy._enter(snapshot)

    assert called == []
    assert strategy.state.trades_today == 0
