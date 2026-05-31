"""Shared account-level equity metrics for live MT5 strategies.

All live strategies should report PnL/drawdown from this single account baseline
instead of seeding their own initial/day equity independently.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - Linux production path has fcntl.
    fcntl = None  # type: ignore[assignment]


DEFAULT_METRICS_PATH = Path(
    "/home/chain4655/Documents/Projects/MT5/auto_quant/state/account_equity_baseline.json"
)


@dataclass(frozen=True)
class AccountMetrics:
    current_day: str
    equity: float
    initial_equity: float
    day_start_equity: float
    max_equity_seen: float
    daily_dd: float
    total_dd: float
    profit_pct: float


class AccountMetricsStore:
    """Atomic JSON store for account-level risk/profit metrics."""

    def __init__(self, path: Path = DEFAULT_METRICS_PATH) -> None:
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + ".lock")

    def update(self, equity: float, day: Optional[str] = None) -> AccountMetrics:
        day_key = day or dt.datetime.now().date().isoformat()
        equity = float(equity)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = self.lock_path.open("a+", encoding="utf-8")
        try:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            data = self._read_unlocked()
            initial = float(data.get("initial_equity") or equity)
            stored_day = str(data.get("current_day") or day_key)
            if stored_day != day_key:
                day_start = equity
                stored_day = day_key
            else:
                day_start = float(data.get("day_start_equity") or equity)
            max_seen = max(float(data.get("max_equity_seen") or equity), equity)
            payload: Dict[str, Any] = {
                "current_day": stored_day,
                "initial_equity": initial,
                "day_start_equity": day_start,
                "max_equity_seen": max_seen,
                "last_equity": equity,
                "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
            }
            self._write_unlocked(payload)
            return self._to_metrics(payload, equity)
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()

    def _read_unlocked(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_unlocked(self, payload: Dict[str, Any]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    @staticmethod
    def _to_metrics(payload: Dict[str, Any], equity: float) -> AccountMetrics:
        initial = max(float(payload["initial_equity"]), 1e-9)
        day_start = max(float(payload["day_start_equity"]), 1e-9)
        max_seen = max(float(payload["max_equity_seen"]), 1e-9)
        return AccountMetrics(
            current_day=str(payload["current_day"]),
            equity=equity,
            initial_equity=initial,
            day_start_equity=day_start,
            max_equity_seen=max_seen,
            daily_dd=1.0 - equity / day_start,
            total_dd=1.0 - equity / max_seen,
            profit_pct=equity / initial - 1.0,
        )
