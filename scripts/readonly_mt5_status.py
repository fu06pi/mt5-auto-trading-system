#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

HOST = "127.0.0.1"
PORT = 18812
IGNORE_PROFIT_ABS_LT = 4.0


def as_dict(obj):
    if obj is None:
        return None
    if hasattr(obj, "_asdict"):
        return obj._asdict()
    if isinstance(obj, dict):
        return obj
    return {name: getattr(obj, name) for name in dir(obj) if not name.startswith("_")}


def main() -> int:
    from pymt5linux import MetaTrader5

    mt5 = MetaTrader5(host=HOST, port=PORT)
    ok = mt5.initialize(path=r"C:\Program Files\MetaTrader 5\terminal64.exe")
    last_error = mt5.last_error()
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "host": HOST,
        "port": PORT,
        "initialize_ok": bool(ok),
        "last_error": list(last_error) if isinstance(last_error, tuple) else str(last_error),
    }
    if not ok:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 2

    try:
        terminal = as_dict(mt5.terminal_info()) or {}
        account = as_dict(mt5.account_info()) or {}
        positions_raw = mt5.positions_get() or []
        positions = [as_dict(p) for p in positions_raw]
        material_positions = [p for p in positions if abs(float(p.get("profit", 0.0))) >= IGNORE_PROFIT_ABS_LT]
        total_profit = sum(float(p.get("profit", 0.0)) for p in material_positions)
        total_volume = sum(float(p.get("volume", 0.0)) for p in material_positions)
        report.update(
            {
                "terminal": {
                    "connected": terminal.get("connected") if terminal else None,
                    "trade_allowed": terminal.get("trade_allowed") if terminal else None,
                    "tradeapi_disabled": terminal.get("tradeapi_disabled") if terminal else None,
                    "name": terminal.get("name") if terminal else None,
                    "path": terminal.get("path") if terminal else None,
                },
                "account": {
                    "login": account.get("login") if account else None,
                    "server": account.get("server") if account else None,
                    "currency": account.get("currency") if account else None,
                    "balance": account.get("balance") if account else None,
                    "equity": account.get("equity") if account else None,
                    "margin": account.get("margin") if account else None,
                    "margin_free": account.get("margin_free") if account else None,
                    "profit": account.get("profit") if account else None,
                    "leverage": account.get("leverage") if account else None,
                },
                "positions_count_raw": len(positions),
                "positions_count_material_abs_profit_gte_4": len(material_positions),
                "positions_material_profit_sum": round(total_profit, 2),
                "positions_material_volume_sum": round(total_volume, 4),
                "positions_material": [
                    {
                        "ticket": p.get("ticket"),
                        "symbol": p.get("symbol"),
                        "type": p.get("type"),
                        "volume": p.get("volume"),
                        "price_open": p.get("price_open"),
                        "price_current": p.get("price_current"),
                        "sl": p.get("sl"),
                        "tp": p.get("tp"),
                        "profit": p.get("profit"),
                        "magic": p.get("magic"),
                        "comment": p.get("comment"),
                    }
                    for p in material_positions
                ],
            }
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
