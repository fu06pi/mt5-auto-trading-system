#!/usr/bin/env python3
"""Build a SQLite database from MT5/strategy logs since a cutoff date.

This is read-only against trading systems: it only scans log files and writes a local
SQLite database/report under Documents/Projects/MT5/reports.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

LOGGER = logging.getLogger("build_strategy_log_db")
DEFAULT_ROOTS = [
    Path("/home/chain4655/Documents/Projects/MT5"),
    Path("/home/chain4655/Documents/Sample/Python"),
]
DEFAULT_OUTPUT = Path(
    "/home/chain4655/Documents/Projects/MT5/reports/strategy_logs_since_2026-03.sqlite"
)
LOG_TS_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}(?:[ T])\d{2}:\d{2}:\d{2}(?:[,.]\d+)?)"
    r"(?:\s+\[(?P<level>[A-Z]+)\])?\s*(?P<msg>.*)$"
)
ISO_START_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*(?P<msg>.*)$")
BAR_TS_RE = re.compile(r"\bBar (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\b")
CLOSED_DEAL_RE = re.compile(
    r"Closed deal sync \| time=(?P<deal_time>\S+) profit=(?P<profit>[-+]?\d+(?:\.\d+)?)"
    r"(?: consecutive_losses=(?P<losses>\d+))?"
)
QUALITY_BLOCK_RE = re.compile(r"Quality filter blocks (?P<side>BUY|SELL): (?P<body>.*)$")
SIGNAL_REVERSAL_RE = re.compile(
    r"Signal reversal TP monitor (?P<status>\w+): position=(?P<position>\w+) adverse=(?P<adverse>\w+)"
    r" count=(?P<count>\d+)/(?P<required>\d+)"
)
KEY_VALUE_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>\[[^\]]*\]|[^\s|,]+)")


@dataclass(frozen=True)
class ParsedLine:
    line_no: int
    raw: str
    log_ts: Optional[str]
    level: Optional[str]
    message: str
    event_type: str
    payload: Dict[str, object]


def parse_timestamp(value: str) -> Optional[dt.datetime]:
    cleaned = value.strip().replace("T", " ").replace(",", ".")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def normalize_ts(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    parsed = parse_timestamp(value)
    if parsed is None:
        return value
    return parsed.isoformat(sep=" ", timespec="seconds")


def infer_strategy_name(path: Path) -> str:
    name = path.name
    for suffix in ("_strategy_supervised.log", "_strategy.log", ".stdout.log", ".log"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def iter_log_files(roots: Iterable[Path]) -> Iterator[Path]:
    skip_parts = {".git", "node_modules", "venv", "venv313", "__pycache__"}
    for root in roots:
        if root.is_file() and root.suffix == ".log":
            yield root
            continue
        if not root.exists():
            continue
        for path in root.rglob("*.log"):
            if skip_parts.intersection(path.parts):
                continue
            if path.stat().st_size <= 0:
                continue
            yield path


def parse_key_values(text: str) -> Dict[str, object]:
    values: Dict[str, object] = {}
    for match in KEY_VALUE_RE.finditer(text):
        key = match.group("key")
        raw_value = match.group("value").strip().strip(",")
        if raw_value in {"True", "False"}:
            values[key] = raw_value == "True"
            continue
        numeric_value = raw_value[:-1] if raw_value.endswith("%") else raw_value
        try:
            if any(ch in numeric_value for ch in (".", "e", "E")):
                values[key] = float(numeric_value)
            else:
                values[key] = int(numeric_value)
            continue
        except ValueError:
            values[key] = raw_value
    return values


def classify_message(message: str) -> Tuple[str, Dict[str, object]]:
    payload = parse_key_values(message)
    event_type = "log"
    if "Risk |" in message:
        event_type = "risk"
    elif "Bar " in message and "signal=" in message:
        event_type = "bar"
        bar_match = BAR_TS_RE.search(message)
        if bar_match:
            payload["bar_time"] = normalize_ts(bar_match.group("ts"))
    elif "Closed deal sync" in message:
        event_type = "closed_deal"
        match = CLOSED_DEAL_RE.search(message)
        if match:
            payload["deal_time"] = normalize_ts(match.group("deal_time"))
            payload["profit"] = float(match.group("profit"))
            if match.group("losses") is not None:
                payload["consecutive_losses"] = int(match.group("losses"))
    elif "Quality filter blocks" in message:
        event_type = "quality_block"
        match = QUALITY_BLOCK_RE.search(message)
        if match:
            payload["side"] = match.group("side")
            payload.update(parse_key_values(match.group("body")))
    elif "Signal reversal TP monitor" in message:
        event_type = "signal_reversal_monitor"
    elif "Auto half-close" in message:
        event_type = "auto_half_close"
    elif "Profit close" in message or "profit-close" in message:
        event_type = "profit_close"
    elif "loss close" in message.lower() or "loss-close" in message.lower():
        event_type = "loss_close"
    elif "order_send" in message or "Order send" in message:
        event_type = "order_send"
    elif "Starting strategy" in message:
        event_type = "starting_strategy"
    elif "Strategy started" in message or "started:" in message:
        event_type = "strategy_started"
    elif "Main loop error" in message or "Traceback" in message or "ERROR" in message:
        event_type = "error"
    elif "paused" in message.lower() or "cooldown" in message.lower():
        event_type = "pause_or_cooldown"
    return event_type, payload


def parse_file(path: Path, cutoff: dt.datetime) -> Iterator[ParsedLine]:
    last_ts: Optional[str] = None
    with path.open("r", errors="replace") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            raw = raw_line.rstrip("\n")
            if not raw:
                continue
            level = None
            message = raw
            log_ts = None
            match = LOG_TS_RE.match(raw) or ISO_START_RE.match(raw)
            if match:
                log_ts = normalize_ts(match.group("ts"))
                last_ts = log_ts
                level = match.groupdict().get("level")
                message = match.groupdict().get("msg") or ""
            else:
                bar_match = BAR_TS_RE.search(raw)
                if bar_match:
                    log_ts = normalize_ts(bar_match.group("ts"))
                else:
                    log_ts = last_ts
            parsed_ts = parse_timestamp(log_ts) if log_ts else None
            if parsed_ts is None or parsed_ts < cutoff:
                continue
            event_type, payload = classify_message(message)
            yield ParsedLine(
                line_no=line_no,
                raw=raw,
                log_ts=normalize_ts(log_ts),
                level=level,
                message=message,
                event_type=event_type,
                payload=payload,
            )


def connect_database(output: Path, reset: bool) -> sqlite3.Connection:
    output.parent.mkdir(parents=True, exist_ok=True)
    if reset and output.exists():
        output.unlink()
    conn = sqlite3.connect(output)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS source_files (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            strategy_name TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            sha1_prefix TEXT NOT NULL,
            first_ts TEXT,
            last_ts TEXT,
            total_lines INTEGER NOT NULL DEFAULT 0,
            imported_lines INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS raw_log_lines (
            id INTEGER PRIMARY KEY,
            file_id INTEGER NOT NULL,
            line_no INTEGER NOT NULL,
            log_ts TEXT,
            level TEXT,
            event_type TEXT NOT NULL,
            message TEXT NOT NULL,
            raw TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            UNIQUE(file_id, line_no),
            FOREIGN KEY(file_id) REFERENCES source_files(id)
        );
        CREATE TABLE IF NOT EXISTS bar_events (
            raw_id INTEGER PRIMARY KEY,
            file_id INTEGER NOT NULL,
            strategy_name TEXT NOT NULL,
            log_ts TEXT,
            bar_time TEXT,
            symbol TEXT,
            signal TEXT,
            htf TEXT,
            htf_comp TEXT,
            session TEXT,
            close REAL,
            atr REAL,
            adx REAL,
            score REAL,
            spread REAL,
            positions INTEGER,
            trades_today INTEGER,
            losses INTEGER,
            payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS risk_events (
            raw_id INTEGER PRIMARY KEY,
            file_id INTEGER NOT NULL,
            strategy_name TEXT NOT NULL,
            log_ts TEXT,
            equity REAL,
            day_dd_pct REAL,
            total_dd_pct REAL,
            profit_pct REAL,
            best_share_pct REAL,
            paused INTEGER,
            payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS closed_deals (
            raw_id INTEGER PRIMARY KEY,
            file_id INTEGER NOT NULL,
            strategy_name TEXT NOT NULL,
            log_ts TEXT,
            deal_time TEXT,
            profit REAL,
            consecutive_losses INTEGER,
            noise_ignored INTEGER NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS event_summary (
            strategy_name TEXT NOT NULL,
            event_type TEXT NOT NULL,
            count INTEGER NOT NULL,
            first_ts TEXT,
            last_ts TEXT,
            PRIMARY KEY(strategy_name, event_type)
        );
        CREATE INDEX IF NOT EXISTS idx_raw_ts ON raw_log_lines(log_ts);
        CREATE INDEX IF NOT EXISTS idx_raw_event ON raw_log_lines(event_type, log_ts);
        CREATE INDEX IF NOT EXISTS idx_bar_strategy_ts ON bar_events(strategy_name, bar_time);
        CREATE INDEX IF NOT EXISTS idx_closed_deals_time ON closed_deals(deal_time);

        DROP VIEW IF EXISTS v_clean_closed_deals;
        CREATE VIEW v_clean_closed_deals AS
        SELECT * FROM closed_deals
        WHERE profit IS NOT NULL AND noise_ignored = 0;

        DROP VIEW IF EXISTS v_dedup_closed_deals_global;
        CREATE VIEW v_dedup_closed_deals_global AS
        SELECT * FROM (
          SELECT cd.*, sf.path,
                 ROW_NUMBER() OVER (
                   PARTITION BY COALESCE(cd.deal_time, cd.log_ts), ROUND(cd.profit, 2)
                   ORDER BY cd.raw_id
                 ) AS rn
          FROM closed_deals cd
          JOIN source_files sf ON sf.id = cd.file_id
          WHERE cd.profit IS NOT NULL AND cd.noise_ignored = 0
        ) WHERE rn = 1;

        DROP VIEW IF EXISTS v_strategy_daily_bars;
        CREATE VIEW v_strategy_daily_bars AS
        SELECT strategy_name,
               substr(COALESCE(bar_time, log_ts), 1, 10) AS day,
               COUNT(*) AS bars,
               SUM(CASE WHEN signal='BUY' THEN 1 ELSE 0 END) AS buy_signals,
               SUM(CASE WHEN signal='SELL' THEN 1 ELSE 0 END) AS sell_signals,
               SUM(CASE WHEN signal='NONE' THEN 1 ELSE 0 END) AS none_signals,
               ROUND(AVG(score), 4) AS avg_score,
               ROUND(AVG(adx), 4) AS avg_adx,
               ROUND(AVG(spread), 2) AS avg_spread
        FROM bar_events
        GROUP BY strategy_name, substr(COALESCE(bar_time, log_ts), 1, 10);

        DROP VIEW IF EXISTS v_strategy_daily_pnl_clean;
        CREATE VIEW v_strategy_daily_pnl_clean AS
        SELECT strategy_name,
               substr(COALESCE(deal_time, log_ts), 1, 10) AS day,
               COUNT(*) AS deals,
               ROUND(SUM(profit), 2) AS net_profit,
               SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) AS wins,
               SUM(CASE WHEN profit < 0 THEN 1 ELSE 0 END) AS losses,
               ROUND(AVG(profit), 2) AS avg_profit
        FROM v_clean_closed_deals
        GROUP BY strategy_name, substr(COALESCE(deal_time, log_ts), 1, 10);
        """
    )
    return conn


def file_sha1_prefix(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def as_float(payload: Dict[str, object], key: str) -> Optional[float]:
    value = payload.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def as_int(payload: Dict[str, object], key: str) -> Optional[int]:
    value = payload.get(key)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default="2026-03-01", help="inclusive cutoff date YYYY-MM-DD")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--root", type=Path, action="append", help="log root; can be repeated")
    parser.add_argument("--reset", action="store_true", help="replace existing database")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    cutoff = dt.datetime.strptime(args.since, "%Y-%m-%d")
    roots = args.root or DEFAULT_ROOTS
    log_files = sorted(set(iter_log_files(roots)))
    conn = connect_database(args.output, reset=args.reset)
    total_imported = 0
    imported_files = 0

    for path in log_files:
        st = path.stat()
        sha1 = file_sha1_prefix(path)
        conn.execute(
            "INSERT OR REPLACE INTO source_files(path, strategy_name, file_size, sha1_prefix) VALUES (?, ?, ?, ?)",
            (str(path), infer_strategy_name(path), st.st_size, sha1),
        )
        file_id = conn.execute("SELECT id FROM source_files WHERE path = ?", (str(path),)).fetchone()[0]
        strategy_name = infer_strategy_name(path)
        imported = 0
        total_lines = 0
        first_ts: Optional[str] = None
        last_ts: Optional[str] = None
        raw_rows: List[Tuple[object, ...]] = []
        bar_rows: List[Tuple[object, ...]] = []
        risk_rows: List[Tuple[object, ...]] = []
        deal_rows: List[Tuple[object, ...]] = []

        with path.open("r", errors="replace") as handle:
            for total_lines, _ in enumerate(handle, 1):
                pass
        for parsed in parse_file(path, cutoff):
            imported += 1
            if first_ts is None:
                first_ts = parsed.log_ts
            last_ts = parsed.log_ts or last_ts
            payload_json = json.dumps(parsed.payload, ensure_ascii=False, sort_keys=True)
            raw_rows.append(
                (
                    file_id,
                    parsed.line_no,
                    parsed.log_ts,
                    parsed.level,
                    parsed.event_type,
                    parsed.message,
                    parsed.raw,
                    payload_json,
                )
            )
        if not raw_rows:
            conn.execute(
                "UPDATE source_files SET first_ts=NULL, last_ts=NULL, total_lines=?, imported_lines=0 WHERE id=?",
                (total_lines, file_id),
            )
            conn.commit()
            continue

        conn.executemany(
            """
            INSERT OR IGNORE INTO raw_log_lines(
                file_id, line_no, log_ts, level, event_type, message, raw, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            raw_rows,
        )
        rows = conn.execute(
            "SELECT id, log_ts, event_type, payload_json FROM raw_log_lines WHERE file_id = ?",
            (file_id,),
        ).fetchall()
        for raw_id, log_ts, event_type, payload_text in rows:
            payload = json.loads(payload_text)
            if event_type == "bar":
                bar_rows.append(
                    (
                        raw_id,
                        file_id,
                        strategy_name,
                        log_ts,
                        payload.get("bar_time"),
                        payload.get("symbol"),
                        payload.get("signal"),
                        payload.get("htf"),
                        payload.get("htf_comp"),
                        payload.get("session"),
                        as_float(payload, "close"),
                        as_float(payload, "atr"),
                        as_float(payload, "adx"),
                        as_float(payload, "score"),
                        as_float(payload, "spread"),
                        as_int(payload, "positions"),
                        as_int(payload, "trades_today"),
                        as_int(payload, "losses"),
                        payload_text,
                    )
                )
            elif event_type == "risk":
                risk_rows.append(
                    (
                        raw_id,
                        file_id,
                        strategy_name,
                        log_ts,
                        as_float(payload, "equity"),
                        as_float(payload, "day_dd"),
                        as_float(payload, "total_dd"),
                        as_float(payload, "profit"),
                        as_float(payload, "best_share"),
                        int(bool(payload.get("paused"))) if "paused" in payload else None,
                        payload_text,
                    )
                )
            elif event_type == "closed_deal":
                profit = as_float(payload, "profit")
                deal_rows.append(
                    (
                        raw_id,
                        file_id,
                        strategy_name,
                        log_ts,
                        payload.get("deal_time"),
                        profit,
                        as_int(payload, "consecutive_losses"),
                        int(abs(profit or 0.0) < 4.0),
                        payload_text,
                    )
                )
        conn.executemany(
            "INSERT OR REPLACE INTO bar_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            bar_rows,
        )
        conn.executemany(
            "INSERT OR REPLACE INTO risk_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            risk_rows,
        )
        conn.executemany(
            "INSERT OR REPLACE INTO closed_deals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            deal_rows,
        )
        conn.execute(
            "UPDATE source_files SET first_ts=?, last_ts=?, total_lines=?, imported_lines=? WHERE id=?",
            (first_ts, last_ts, total_lines, imported, file_id),
        )
        conn.commit()
        total_imported += imported
        imported_files += 1
        LOGGER.info("imported %-45s lines=%d", path.name, imported)

    conn.execute("DELETE FROM event_summary")
    conn.execute(
        """
        INSERT INTO event_summary(strategy_name, event_type, count, first_ts, last_ts)
        SELECT sf.strategy_name, r.event_type, COUNT(*), MIN(r.log_ts), MAX(r.log_ts)
        FROM raw_log_lines r
        JOIN source_files sf ON sf.id = r.file_id
        GROUP BY sf.strategy_name, r.event_type
        """
    )
    conn.commit()
    conn.close()
    LOGGER.info("done db=%s files=%d imported_lines=%d", args.output, imported_files, total_imported)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
