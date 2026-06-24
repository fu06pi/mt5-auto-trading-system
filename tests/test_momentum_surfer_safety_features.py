import datetime as dt
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


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
    DEAL_ENTRY_OUT = 1
    DEAL_ENTRY_OUT_BY = 3

    def __init__(self, *args, **kwargs):
        pass


def load_strategy_module():
    fake_module = types.ModuleType("pymt5linux")
    fake_module.MetaTrader5 = FakeMetaTrader5
    sys.modules["pymt5linux"] = fake_module

    path = Path("/home/chain4655/Documents/Projects/MT5/strategies/mt5_xauusd_momentum_surfer_strategy.py")
    spec = importlib.util.spec_from_file_location("momentum_surfer_strategy_safety", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class HistoryMT5(FakeMetaTrader5):
    def __init__(self, deals):
        self.deals = deals
        self.query_from = None
        self.query_to = None

    def history_deals_get(self, from_time, to_time):
        self.query_from = from_time
        self.query_to = to_time
        return self.deals

    def last_error(self):
        return (-10004, "No IPC connection")


class SymbolMT5(FakeMetaTrader5):
    def __init__(self, info_results):
        self.info_results = list(info_results)
        self.select_calls = 0

    def symbol_select(self, symbol, enable):
        self.select_calls += 1
        return True

    def symbol_info(self, symbol):
        if self.info_results:
            return self.info_results.pop(0)
        return None

    def last_error(self):
        return (-10004, "No IPC connection")


def make_strategy(module, deals):
    strategy = module.XAUUSDMomentumSurferStrategy.__new__(module.XAUUSDMomentumSurferStrategy)
    strategy.mt5 = HistoryMT5(deals)
    strategy.config = SimpleNamespace(
        symbol="XAUUSD",
        magic=210511,
        loss_cooldown_losses=2,
        loss_cooldown_minutes=15,
        loss_close_pause_minutes=30,
        profit_close_usd=0.0,
        profit_close_pause_minutes=0,
    )
    strategy.state = module.StrategyState()
    strategy._logs = []
    strategy._log = lambda message, *args: strategy._logs.append(message % args if args else message)
    strategy._save_state = lambda: None
    return strategy


def deal(ticket, timestamp, profit, magic=210511, symbol="XAUUSD"):
    return SimpleNamespace(
        ticket=ticket,
        time=timestamp,
        profit=profit,
        commission=0.0,
        swap=0.0,
        magic=magic,
        symbol=symbol,
        entry=FakeMetaTrader5.DEAL_ENTRY_OUT,
    )


def test_closed_deal_sync_uses_ticket_cursor_for_same_second_deals():
    module = load_strategy_module()
    ts = int(dt.datetime(2026, 6, 21, 12, 0, tzinfo=dt.UTC).timestamp())
    strategy = make_strategy(module, [deal(10, ts, -20.0), deal(11, ts, -30.0)])

    strategy._sync_closed_trades()

    assert strategy.state.consecutive_losses == 2
    assert strategy.state.last_processed_deal_ticket == 11
    assert strategy.state.loss_cooldown_until is not None
    assert strategy.state.loss_pause_until is not None


def test_closed_deal_sync_filters_foreign_and_noise_deals():
    module = load_strategy_module()
    ts = int(dt.datetime(2026, 6, 21, 12, 0, tzinfo=dt.UTC).timestamp())
    strategy = make_strategy(
        module,
        [
            deal(10, ts, -100.0, magic=0),
            deal(11, ts + 1, -3.0),
            deal(12, ts + 2, -25.0),
        ],
    )

    strategy._sync_closed_trades()

    assert strategy.state.consecutive_losses == 1
    assert strategy.state.last_processed_deal_ticket == 12
    assert strategy.state.last_close_profit == -25.0


def test_history_query_uses_local_naive_window_but_state_uses_utc_chart_time():
    module = load_strategy_module()
    ts = int(dt.datetime(2026, 6, 21, 12, 0, tzinfo=dt.UTC).timestamp())
    strategy = make_strategy(module, [deal(10, ts, -20.0)])

    strategy._sync_closed_trades()

    assert strategy.mt5.query_to.tzinfo is None
    assert strategy.state.last_processed_deal_time == "2026-06-21T12:00:00"


def test_prepare_symbol_retries_and_caches_symbol_info():
    module = load_strategy_module()
    info = SimpleNamespace(digits=2, point=0.01, volume_min=0.01, volume_step=0.01)
    strategy = module.XAUUSDMomentumSurferStrategy.__new__(module.XAUUSDMomentumSurferStrategy)
    strategy.mt5 = SymbolMT5([None, None, info])
    strategy.symbol = "XAUUSD"
    strategy._cached_symbol_info = None
    strategy._cached_point = None
    strategy._cached_digits = None
    strategy._logs = []
    strategy._log = lambda message, *args: strategy._logs.append(message % args if args else message)

    strategy._prepare_symbol()

    assert strategy.mt5.select_calls == 3
    assert strategy._point() == 0.01
    assert strategy._digits() == 2
    assert strategy._symbol_info() is info


def test_new_cli_safety_flags_exist():
    module = load_strategy_module()
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--max-lots-per-order",
            "0.3",
            "--loss-close-pause-minutes",
            "45",
            "--profit-close-usd",
            "200",
            "--profit-close-pause-minutes",
            "20",
            "--order-comment",
            "surfer-test",
        ]
    )

    assert args.max_lots_per_order == 0.3
    assert args.loss_close_pause_minutes == 45
    assert args.profit_close_usd == 200.0
    assert args.profit_close_pause_minutes == 20
    assert args.order_comment == "surfer-test"
