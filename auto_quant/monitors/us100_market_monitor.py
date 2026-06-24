#!/usr/bin/env python3
"""Read-only MT5 market monitor for US100.cash.

Logs tick/account/position snapshots. It does not send orders.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

from pymt5linux import MetaTrader5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only MT5 symbol monitor")
    parser.add_argument("--symbol", default="US100.cash")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18812)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument(
        "--log-file",
        default="/home/chain4655/Documents/Projects/MT5/auto_quant/logs/us100_market_watch.log",
    )
    return parser.parse_args()


def setup_logging(log_file: str) -> None:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )


def main() -> int:
    args = parse_args()
    setup_logging(args.log_file)
    stop = False

    def _stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    logging.info("US100 monitor starting symbol=%s host=%s port=%s interval=%ss", args.symbol, args.host, args.port, args.interval)

    while not stop:
        mt5 = MetaTrader5(host=args.host, port=args.port)
        try:
            ok = mt5.initialize()
            if not ok:
                logging.warning("initialize=False last_error=%s", mt5.last_error())
                time.sleep(args.interval)
                continue
            selected = mt5.symbol_select(args.symbol, True)
            info = mt5.symbol_info(args.symbol)
            tick = mt5.symbol_info_tick(args.symbol)
            account = mt5.account_info()
            positions = mt5.positions_get(symbol=args.symbol)
            orders = mt5.orders_get(symbol=args.symbol)
            if info is None or tick is None:
                logging.warning("symbol unavailable symbol=%s selected=%s last_error=%s", args.symbol, selected, mt5.last_error())
            else:
                spread = None
                if getattr(info, "point", 0):
                    spread = (tick.ask - tick.bid) / info.point
                logging.info(
                    "symbol=%s selected=%s bid=%.2f ask=%.2f spread_points=%s visible=%s trade_mode=%s account_equity=%s positions=%s orders=%s",
                    args.symbol,
                    selected,
                    tick.bid,
                    tick.ask,
                    None if spread is None else round(spread, 1),
                    getattr(info, "visible", None),
                    getattr(info, "trade_mode", None),
                    None if account is None else round(account.equity, 2),
                    None if positions is None else len(positions),
                    None if orders is None else len(orders),
                )
        except (OSError, RuntimeError, ValueError) as exc:
            logging.exception("monitor iteration failed: %s", exc)
        finally:
            try:
                mt5.shutdown()
            except (OSError, RuntimeError, ValueError):
                logging.exception("shutdown failed")
        time.sleep(args.interval)

    logging.info("US100 monitor stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
