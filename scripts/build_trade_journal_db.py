#!/usr/bin/env python3
"""Build a read-only MT5 trade journal database.

Records broker history plus nearby strategy-log context so each position has:
entry time/price, exit time/price, side, volume, PnL, magic/comment, and inferred
entry/exit reasons.

This script does not edit active_plan.json, does not restart MT5, and does not send orders.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT = Path("/home/chain4655/Documents/Projects/MT5")
DEFAULT_DB = PROJECT / "reports" / "mt5_trade_journal.sqlite"
DEFAULT_CSV = PROJECT / "reports" / "mt5_trade_journal.csv"
PYMT5_SITE = Path("/home/chain4655/.openharness-venv/lib64/python3.14/site-packages")
if str(PYMT5_SITE) not in sys.path:
    sys.path.insert(0, str(PYMT5_SITE))

try:
    from pymt5linux import MetaTrader5  # type: ignore
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(f"pymt5linux import failed: {exc}") from exc

LOGGER = logging.getLogger("build_trade_journal_db")

DEAL_TYPE = {0: "BUY", 1: "SELL", 2: "BALANCE", 3: "CREDIT", 4: "CHARGE", 5: "CORRECTION", 6: "BONUS", 7: "COMMISSION", 8: "COMMISSION_DAILY", 9: "COMMISSION_MONTHLY", 10: "COMMISSION_AGENT_DAILY", 11: "COMMISSION_AGENT_MONTHLY", 12: "INTEREST", 13: "BUY_CANCELED", 14: "SELL_CANCELED", 15: "DIVIDEND", 16: "DIVIDEND_FRANKED", 17: "TAX"}
DEAL_ENTRY = {0: "IN", 1: "OUT", 2: "INOUT", 3: "OUT_BY"}
DEAL_REASON = {
    0: "CLIENT",
    1: "MOBILE",
    2: "WEB",
    3: "EXPERT",
    4: "SL",
    5: "TP",
    6: "SO",
    7: "ROLLOVER",
    8: "VMARGIN",
    9: "SPLIT",
}

TS_RE = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2}(?:[ T])\d{2}:\d{2}:\d{2})(?:[,\.](?P<ms>\d+))?")
BAR_RE = re.compile(r"Bar (?P<bar_time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| (?P<body>.*)")
KEY_VALUE_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>\[[^\]]*\]|[^\s|,]+)")


@dataclass
class LogEvent:
    log_ts: Optional[str]
    event_type: str
    strategy_name: str
    source_file: str
    line_no: int
    message: str
    payload: Dict[str, Any]


def parse_dt(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    text = str(value).strip().replace("T", " ").replace(",", ".")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text[:26], fmt)
        except ValueError:
            continue
    return None


def iso_from_epoch(value: Any, millis: bool = False) -> Optional[str]:
    if value in (None, 0, ""):
        return None
    try:
        seconds = int(value) / (1000 if millis else 1)
        return datetime.fromtimestamp(seconds).isoformat(sep=" ", timespec="seconds")
    except (OSError, TypeError, ValueError):
        return None


def normalize_ts(value: Any) -> Optional[str]:
    parsed = parse_dt(value)
    if parsed is None:
        return None
    return parsed.isoformat(sep=" ", timespec="seconds")


def rowdict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "_asdict"):
        data = dict(obj._asdict())
    elif isinstance(obj, dict):
        data = dict(obj)
    else:
        data = {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")}
    for key in ("time", "time_setup", "time_done", "expiration"):
        if key in data:
            data[f"{key}_iso"] = iso_from_epoch(data.get(key), millis=False)
    for key in ("time_msc", "time_setup_msc", "time_done_msc"):
        if key in data:
            data[f"{key}_iso"] = iso_from_epoch(data.get(key), millis=True)
    if "type" in data:
        type_value = data.get("type")
        data["type_name"] = DEAL_TYPE.get(type_value, str(type_value)) if isinstance(type_value, int) else str(type_value)
    if "entry" in data:
        entry_value = data.get("entry")
        data["entry_name"] = DEAL_ENTRY.get(entry_value, str(entry_value)) if isinstance(entry_value, int) else str(entry_value)
    if "reason" in data:
        reason_value = data.get("reason")
        data["reason_name"] = DEAL_REASON.get(reason_value, str(reason_value)) if isinstance(reason_value, int) else str(reason_value)
    return data


def parse_key_values(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for match in KEY_VALUE_RE.finditer(text):
        key = match.group("key")
        raw = match.group("value").strip().strip(",")
        if raw in {"True", "False"}:
            out[key] = raw == "True"
            continue
        numeric = raw[:-1] if raw.endswith("%") else raw
        try:
            out[key] = float(numeric) if any(c in numeric for c in ".eE") else int(numeric)
        except ValueError:
            out[key] = raw
    return out


def infer_strategy_name(path: Path) -> str:
    name = path.name
    for suffix in ("_strategy_supervised.log", "_strategy.log", ".stdout.log", ".log"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def discover_logs() -> List[Path]:
    roots = [PROJECT, Path("/home/chain4655/Documents/Sample/Python")]
    logs: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.log"):
            if any(part in {".git", "__pycache__", "node_modules", "venv", "venv313"} for part in path.parts):
                continue
            if path.stat().st_size > 0:
                logs.append(path)
    # include common supervisor stdout files without .log only when present
    for path in [PROJECT / "xauusd_asian_reversal_supervisor.out"]:
        if path.exists() and path.stat().st_size > 0:
            logs.append(path)
    return sorted(set(logs))


def classify_log_line(path: Path, line_no: int, line: str) -> Optional[LogEvent]:
    match = TS_RE.search(line[:40])
    log_ts = normalize_ts(match.group("ts")) if match else None
    message = line
    if match and line.startswith(match.group("ts")):
        message = line[match.end() :].strip()
        message = re.sub(r"^\[[A-Z]+\]\s*", "", message)
    payload = parse_key_values(message)
    event_type = "log"
    bar_match = BAR_RE.search(message)
    if bar_match and "signal=" in message:
        event_type = "bar_signal"
        payload.update(parse_key_values(bar_match.group("body")))
        payload["bar_time"] = normalize_ts(bar_match.group("bar_time"))
    elif "Closed deal sync" in message:
        event_type = "closed_deal_sync"
    elif "Quality filter blocks" in message:
        event_type = "quality_block"
    elif "order_send" in message or "Order send" in message or "ORDER_SEND" in message or "retcode" in message:
        event_type = "order_event"
    elif "Signal reversal" in message:
        event_type = "signal_reversal_exit"
    elif "Profit close" in message or "profit-close" in message:
        event_type = "profit_close_exit"
    elif "loss close" in message.lower() or "loss-close" in message.lower():
        event_type = "loss_close_exit"
    elif "max hold" in message.lower() or "max_hold" in message.lower():
        event_type = "max_hold_exit"
    elif "force-close" in message.lower() or "force close" in message.lower():
        event_type = "force_close_exit"
    elif "Bar " in message and "signal=" in message:
        event_type = "bar_signal"
    else:
        return None
    return LogEvent(
        log_ts=log_ts,
        event_type=event_type,
        strategy_name=infer_strategy_name(path),
        source_file=str(path),
        line_no=line_no,
        message=message[:4000],
        payload=payload,
    )


def collect_log_events(since: datetime) -> List[LogEvent]:
    events: List[LogEvent] = []
    for path in discover_logs():
        try:
            with path.open("r", errors="replace") as handle:
                for line_no, raw in enumerate(handle, 1):
                    line = raw.rstrip("\n")
                    if not any(token in line for token in ("Bar ", "signal=", "Closed deal", "order_send", "Order send", "ORDER_SEND", "retcode", "Signal reversal", "Profit close", "loss close", "loss-close", "max hold", "force-close", "Quality filter")):
                        continue
                    event = classify_log_line(path, line_no, line)
                    if event is None:
                        continue
                    when = parse_dt(event.payload.get("bar_time")) or parse_dt(event.log_ts)
                    if when is not None and when >= since:
                        events.append(event)
        except OSError as exc:
            LOGGER.warning("skip unreadable log %s: %s", path, exc)
    return events


def option_value(cmd: List[str], flag: str) -> Optional[str]:
    try:
        idx = cmd.index(flag)
    except ValueError:
        return None
    if idx + 1 < len(cmd):
        return cmd[idx + 1]
    return None


def load_plan_map() -> Dict[int, Dict[str, Any]]:
    path = PROJECT / "auto_quant" / "active_plan.json"
    if not path.exists():
        return {}
    plan = json.loads(path.read_text(encoding="utf-8"))
    commands: List[Tuple[str, List[str]]] = []
    if isinstance(plan.get("cmd"), list):
        commands.append(("primary", plan["cmd"]))
    for item in plan.get("complementary_strategies", []) or []:
        if isinstance(item, dict) and isinstance(item.get("cmd"), list):
            commands.append((item.get("name") or item.get("id") or "complement", item["cmd"]))
    out: Dict[int, Dict[str, Any]] = {}
    for name, cmd in commands:
        magic_text = option_value(cmd, "--magic")
        if not magic_text:
            continue
        try:
            magic = int(magic_text)
        except ValueError:
            continue
        out[magic] = {
            "plan_name": name,
            "strategy_file": next((x for x in cmd if x.endswith(".py")), ""),
            "symbol": option_value(cmd, "--symbol") or "XAUUSD",
            "timeframe": option_value(cmd, "--timeframe"),
            "comment": option_value(cmd, "--order-comment"),
            "log_file": option_value(cmd, "--log-file"),
        }
    return out


def net_of_deal(deal: Dict[str, Any]) -> float:
    return sum(float(deal.get(k) or 0.0) for k in ("profit", "commission", "swap", "fee"))


def weighted_price(rows: List[Dict[str, Any]]) -> Optional[float]:
    total_volume = sum(float(r.get("volume") or 0.0) for r in rows)
    if total_volume <= 0:
        return None
    return round(sum(float(r.get("price") or 0.0) * float(r.get("volume") or 0.0) for r in rows) / total_volume, 5)


def nearest_signal(events: List[LogEvent], when: Optional[str], side: str, symbol: str, max_minutes: int = 180) -> Optional[LogEvent]:
    target = parse_dt(when)
    if target is None:
        return None
    best: Optional[Tuple[float, LogEvent]] = None
    for event in events:
        if event.event_type != "bar_signal":
            continue
        payload = event.payload
        signal = str(payload.get("signal") or "")
        if signal != side:
            continue
        event_symbol = str(payload.get("symbol") or "")
        if event_symbol and event_symbol != symbol:
            continue
        event_time = parse_dt(payload.get("bar_time")) or parse_dt(event.log_ts)
        if event_time is None or event_time > target:
            continue
        delta_minutes = (target - event_time).total_seconds() / 60
        if delta_minutes > max_minutes:
            continue
        if best is None or delta_minutes < best[0]:
            best = (delta_minutes, event)
    return best[1] if best else None


def nearest_exit_event(events: List[LogEvent], when: Optional[str], max_minutes: int = 180) -> Optional[LogEvent]:
    target = parse_dt(when)
    if target is None:
        return None
    exit_types = {"signal_reversal_exit", "profit_close_exit", "loss_close_exit", "max_hold_exit", "force_close_exit", "closed_deal_sync", "order_event"}
    best: Optional[Tuple[float, LogEvent]] = None
    for event in events:
        if event.event_type not in exit_types:
            continue
        event_time = parse_dt(event.payload.get("deal_time")) or parse_dt(event.log_ts)
        if event_time is None:
            continue
        delta_minutes = abs((event_time - target).total_seconds()) / 60
        if delta_minutes > max_minutes:
            continue
        if best is None or delta_minutes < best[0]:
            best = (delta_minutes, event)
    return best[1] if best else None


def infer_entry_reason(position: Dict[str, Any], signal: Optional[LogEvent], plan_info: Dict[str, Any]) -> str:
    parts = []
    if plan_info.get("plan_name"):
        parts.append(f"plan={plan_info['plan_name']}")
    if position.get("comment_open"):
        parts.append(f"comment={position['comment_open']}")
    if signal:
        p = signal.payload
        bits = [f"signal={p.get('signal')}"]
        for key in ("source", "score", "htf", "htf_comp", "session", "chop_reason", "fb_reason"):
            if p.get(key) not in (None, ""):
                bits.append(f"{key}={p.get(key)}")
        parts.append("nearest_log: " + " ".join(bits))
    if not parts:
        parts.append("broker_history_only: no matching strategy signal log found")
    return " | ".join(parts)


def infer_exit_reason(exit_deals: List[Dict[str, Any]], exit_event: Optional[LogEvent]) -> str:
    reasons = sorted({str(d.get("reason_name") or d.get("reason")) for d in exit_deals if d.get("reason") is not None})
    parts = []
    if reasons:
        parts.append("broker_reason=" + "/".join(reasons))
    comments = sorted({str(d.get("comment")) for d in exit_deals if d.get("comment")})
    if comments:
        parts.append("comment=" + "/".join(comments[:3]))
    if exit_event:
        parts.append(f"nearest_log={exit_event.event_type}: {exit_event.message[:240]}")
    if not parts:
        parts.append("open_or_unknown")
    return " | ".join(parts)


def fetch_mt5_history(host: str, port: int, since: datetime) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    mt5 = MetaTrader5(host=host, port=port)
    if not mt5.initialize():
        raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")
    account = rowdict(mt5.account_info() or {})
    terminal = rowdict(mt5.terminal_info() or {})
    now = datetime.now()
    deals = [rowdict(x) for x in (mt5.history_deals_get(since, now) or [])]
    orders = [rowdict(x) for x in (mt5.history_orders_get(since, now) or [])]
    positions = [rowdict(x) for x in (mt5.positions_get() or [])]
    mt5.shutdown()
    return account, terminal, deals, orders, positions


def build_trade_rows(deals: List[Dict[str, Any]], events: List[LogEvent], plan_map: Dict[int, Dict[str, Any]], account: Dict[str, Any]) -> List[Dict[str, Any]]:
    trade_deals = [d for d in deals if d.get("type") in (0, 1)]
    by_position: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for deal in trade_deals:
        pos_id = deal.get("position_id") or deal.get("order") or deal.get("ticket")
        by_position[pos_id].append(deal)
    rows: List[Dict[str, Any]] = []
    for pos_id, group in by_position.items():
        group = sorted(group, key=lambda d: (d.get("time") or 0, d.get("ticket") or 0))
        entries = [d for d in group if d.get("entry") == 0]
        exits = [d for d in group if d.get("entry") in (1, 3)]
        first = entries[0] if entries else group[0]
        last = exits[-1] if exits else group[-1]
        type_value = first.get("type")
        side = first.get("type_name") or (DEAL_TYPE.get(type_value, "UNKNOWN") if isinstance(type_value, int) else "UNKNOWN")
        symbol = str(first.get("symbol") or "")
        magic = int(first.get("magic") or 0)
        volume_in = round(sum(float(d.get("volume") or 0.0) for d in entries), 4)
        volume_out = round(sum(float(d.get("volume") or 0.0) for d in exits), 4)
        net = round(sum(net_of_deal(d) for d in group), 2)
        open_time = first.get("time_iso") or first.get("time_msc_iso")
        close_time = last.get("time_iso") or last.get("time_msc_iso") if exits else None
        signal = nearest_signal(events, open_time, str(side), symbol)
        exit_event = nearest_exit_event(events, close_time) if exits else None
        plan_info = plan_map.get(magic, {})
        row = {
            "account_login": account.get("login"),
            "account_server": account.get("server"),
            "position_id": pos_id,
            "status": "closed" if exits and abs(volume_in - volume_out) < 1e-9 else "open_or_partial",
            "symbol": symbol,
            "side": side,
            "volume_in": volume_in,
            "volume_out": volume_out,
            "open_time": open_time,
            "open_price": weighted_price(entries) if entries else first.get("price"),
            "close_time": close_time,
            "close_price": weighted_price(exits) if exits else None,
            "net_profit": net,
            "gross_profit": round(sum(float(d.get("profit") or 0.0) for d in group), 2),
            "commission": round(sum(float(d.get("commission") or 0.0) for d in group), 2),
            "swap": round(sum(float(d.get("swap") or 0.0) for d in group), 2),
            "fee": round(sum(float(d.get("fee") or 0.0) for d in group), 2),
            "magic": magic,
            "strategy_name": plan_info.get("plan_name") or "unknown_or_historical",
            "strategy_file": plan_info.get("strategy_file") or "",
            "comment_open": first.get("comment"),
            "entry_reason": infer_entry_reason({"comment_open": first.get("comment")}, signal, plan_info),
            "exit_reason": infer_exit_reason(exits, exit_event) if exits else "open_position",
            "entry_context_json": json.dumps(signal.payload if signal else {}, ensure_ascii=False, sort_keys=True),
            "exit_context_json": json.dumps(exit_event.payload if exit_event else {}, ensure_ascii=False, sort_keys=True),
            "deal_tickets": ";".join(str(d.get("ticket")) for d in group),
            "order_ids": ";".join(str(d.get("order")) for d in group),
            "noise_ignored": int(abs(net) < 4.0),
        }
        rows.append(row)
    rows.sort(key=lambda r: (r.get("close_time") or r.get("open_time") or "", str(r.get("position_id"))))
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def connect_db(path: Path, reset: bool) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if reset and path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS import_runs (
            id INTEGER PRIMARY KEY,
            generated_at TEXT NOT NULL,
            since TEXT NOT NULL,
            account_login INTEGER,
            account_server TEXT,
            raw_deals INTEGER NOT NULL,
            raw_orders INTEGER NOT NULL,
            log_events INTEGER NOT NULL,
            trade_records INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS account_snapshot (
            run_id INTEGER PRIMARY KEY,
            payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS raw_deals (
            ticket INTEGER PRIMARY KEY,
            position_id INTEGER,
            order_id INTEGER,
            time_iso TEXT,
            symbol TEXT,
            type_name TEXT,
            entry_name TEXT,
            reason_name TEXT,
            volume REAL,
            price REAL,
            profit REAL,
            commission REAL,
            swap REAL,
            fee REAL,
            magic INTEGER,
            comment TEXT,
            payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS raw_orders (
            ticket INTEGER PRIMARY KEY,
            position_id INTEGER,
            time_setup_iso TEXT,
            time_done_iso TEXT,
            symbol TEXT,
            type INTEGER,
            state INTEGER,
            volume_initial REAL,
            volume_current REAL,
            price_open REAL,
            price_current REAL,
            sl REAL,
            tp REAL,
            magic INTEGER,
            comment TEXT,
            payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS log_events (
            id INTEGER PRIMARY KEY,
            log_ts TEXT,
            event_type TEXT NOT NULL,
            strategy_name TEXT,
            source_file TEXT,
            line_no INTEGER,
            message TEXT,
            payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS trade_journal (
            position_id TEXT PRIMARY KEY,
            account_login INTEGER,
            account_server TEXT,
            status TEXT,
            symbol TEXT,
            side TEXT,
            volume_in REAL,
            volume_out REAL,
            open_time TEXT,
            open_price REAL,
            close_time TEXT,
            close_price REAL,
            net_profit REAL,
            gross_profit REAL,
            commission REAL,
            swap REAL,
            fee REAL,
            magic INTEGER,
            strategy_name TEXT,
            strategy_file TEXT,
            comment_open TEXT,
            entry_reason TEXT,
            exit_reason TEXT,
            entry_context_json TEXT,
            exit_context_json TEXT,
            deal_tickets TEXT,
            order_ids TEXT,
            noise_ignored INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_trade_journal_time ON trade_journal(open_time, close_time);
        CREATE INDEX IF NOT EXISTS idx_trade_journal_magic ON trade_journal(magic);
        CREATE INDEX IF NOT EXISTS idx_trade_journal_symbol ON trade_journal(symbol);
        CREATE INDEX IF NOT EXISTS idx_log_events_time_type ON log_events(log_ts, event_type);

        DROP VIEW IF EXISTS v_trade_journal_clean;
        CREATE VIEW v_trade_journal_clean AS
        SELECT * FROM trade_journal WHERE noise_ignored = 0;

        DROP VIEW IF EXISTS v_trade_journal_daily_pnl;
        CREATE VIEW v_trade_journal_daily_pnl AS
        SELECT substr(COALESCE(close_time, open_time), 1, 10) AS day,
               account_login, account_server, strategy_name, symbol,
               COUNT(*) AS trades,
               ROUND(SUM(net_profit), 2) AS net_profit,
               SUM(CASE WHEN net_profit > 0 THEN 1 ELSE 0 END) AS wins,
               SUM(CASE WHEN net_profit < 0 THEN 1 ELSE 0 END) AS losses
        FROM v_trade_journal_clean
        GROUP BY substr(COALESCE(close_time, open_time), 1, 10), account_login, account_server, strategy_name, symbol;
        """
    )
    return conn


def persist(conn: sqlite3.Connection, since: datetime, account: Dict[str, Any], terminal: Dict[str, Any], deals: List[Dict[str, Any]], orders: List[Dict[str, Any]], events: List[LogEvent], trade_rows: List[Dict[str, Any]]) -> int:
    generated_at = datetime.now().isoformat(sep=" ", timespec="seconds")
    cur = conn.execute(
        "INSERT INTO import_runs(generated_at, since, account_login, account_server, raw_deals, raw_orders, log_events, trade_records) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (generated_at, since.isoformat(sep=" ", timespec="seconds"), account.get("login"), account.get("server"), len(deals), len(orders), len(events), len(trade_rows)),
    )
    if cur.lastrowid is None:
        raise RuntimeError("failed to create import_runs row")
    run_id = int(cur.lastrowid)
    conn.execute("INSERT OR REPLACE INTO account_snapshot(run_id, payload_json) VALUES (?, ?)", (run_id, json.dumps({"account": account, "terminal": terminal}, ensure_ascii=False, sort_keys=True)))
    conn.executemany(
        "INSERT OR REPLACE INTO raw_deals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                d.get("ticket"), d.get("position_id"), d.get("order"), d.get("time_iso") or d.get("time_msc_iso"), d.get("symbol"), d.get("type_name"), d.get("entry_name"), d.get("reason_name"), d.get("volume"), d.get("price"), d.get("profit"), d.get("commission"), d.get("swap"), d.get("fee"), d.get("magic"), d.get("comment"), json.dumps(d, ensure_ascii=False, sort_keys=True),
            )
            for d in deals
            if d.get("ticket") is not None
        ],
    )
    conn.executemany(
        "INSERT OR REPLACE INTO raw_orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                o.get("ticket"), o.get("position_id"), o.get("time_setup_iso"), o.get("time_done_iso"), o.get("symbol"), o.get("type"), o.get("state"), o.get("volume_initial"), o.get("volume_current"), o.get("price_open"), o.get("price_current"), o.get("sl"), o.get("tp"), o.get("magic"), o.get("comment"), json.dumps(o, ensure_ascii=False, sort_keys=True),
            )
            for o in orders
            if o.get("ticket") is not None
        ],
    )
    conn.execute("DELETE FROM log_events")
    conn.executemany(
        "INSERT INTO log_events(log_ts, event_type, strategy_name, source_file, line_no, message, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(e.log_ts, e.event_type, e.strategy_name, e.source_file, e.line_no, e.message, json.dumps(e.payload, ensure_ascii=False, sort_keys=True)) for e in events],
    )
    conn.executemany(
        "INSERT OR REPLACE INTO trade_journal VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                str(r.get("position_id")), r.get("account_login"), r.get("account_server"), r.get("status"), r.get("symbol"), r.get("side"), r.get("volume_in"), r.get("volume_out"), r.get("open_time"), r.get("open_price"), r.get("close_time"), r.get("close_price"), r.get("net_profit"), r.get("gross_profit"), r.get("commission"), r.get("swap"), r.get("fee"), r.get("magic"), r.get("strategy_name"), r.get("strategy_file"), r.get("comment_open"), r.get("entry_reason"), r.get("exit_reason"), r.get("entry_context_json"), r.get("exit_context_json"), r.get("deal_tickets"), r.get("order_ids"), r.get("noise_ignored"),
            )
            for r in trade_rows
        ],
    )
    conn.commit()
    return run_id


def write_readme(path: Path, db_path: Path, csv_path: Path) -> None:
    content = f"""# MT5 Trade Journal Database

Read-only trade-record system for MT5.

## Purpose

Stores each broker position as one journal row with:

- entry time / entry price
- exit time / exit price
- symbol, side, volume, magic, order comment
- net PnL / commission / swap / fee
- inferred entry reason from nearest strategy `Bar ... signal=BUY/SELL` log
- inferred exit reason from MT5 deal reason plus nearest exit-related log line

## Files

- SQLite DB: `{db_path}`
- CSV export: `{csv_path}`
- Builder: `{PROJECT / 'scripts' / 'build_trade_journal_db.py'}`

## Rebuild

```bash
cd {PROJECT}
/home/chain4655/.openharness-venv/bin/python scripts/build_trade_journal_db.py --since 2026-03-01 --reset
```

## Useful SQL

```sql
SELECT open_time, close_time, symbol, side, volume_in, open_price, close_price,
       net_profit, magic, strategy_name, entry_reason, exit_reason
FROM trade_journal
ORDER BY COALESCE(close_time, open_time) DESC
LIMIT 20;

SELECT * FROM v_trade_journal_daily_pnl ORDER BY day DESC;

SELECT symbol, magic, strategy_name, COUNT(*) trades, ROUND(SUM(net_profit),2) pnl
FROM v_trade_journal_clean
GROUP BY symbol, magic, strategy_name
ORDER BY pnl DESC;
```

## Caveats

- Broker history is authoritative for prices/PnL, but only for the account currently logged into the MT5 terminal/bridge.
- Entry/exit reasons are inferred from strategy logs and may be empty for manual trades, old logs, account switches, or missing log lines.
- Rows with `abs(net_profit) < 4` are marked `noise_ignored=1` for this user's normal MT5 reporting preference.
- This script does not modify live strategy config, does not restart processes, and does not place orders.
"""
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18812)
    parser.add_argument("--since", default="2026-03-01", help="inclusive local-naive start date YYYY-MM-DD")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--reset", action="store_true", help="replace existing SQLite database")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    since = datetime.strptime(args.since, "%Y-%m-%d")
    plan_map = load_plan_map()
    events = collect_log_events(since)
    account, terminal, deals, orders, _positions = fetch_mt5_history(args.host, args.port, since)
    trade_rows = build_trade_rows(deals, events, plan_map, account)
    conn = connect_db(args.db, args.reset)
    run_id = persist(conn, since, account, terminal, deals, orders, events, trade_rows)
    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    conn.close()
    write_csv(args.csv, trade_rows)
    readme = args.db.with_name("mt5_trade_journal_README.md")
    write_readme(readme, args.db, args.csv)
    summary = {
        "run_id": run_id,
        "db": str(args.db),
        "csv": str(args.csv),
        "readme": str(readme),
        "integrity_check": integrity,
        "account": {k: account.get(k) for k in ("login", "server", "balance", "equity", "trade_allowed", "trade_expert")},
        "counts": {"raw_deals": len(deals), "raw_orders": len(orders), "log_events": len(events), "trade_records": len(trade_rows), "clean_trade_records": sum(1 for r in trade_rows if not r.get("noise_ignored"))},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if integrity == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
