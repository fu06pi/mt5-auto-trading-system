from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class StrategySpec:
    name: str
    cmd: List[str]
    log_file: Path


@dataclass(frozen=True)
class SupervisorConfig:
    host: str = "127.0.0.1"
    port: int = 18812
    wineprefix: str = "/home/chain4655/.mt5"
    win_python: str = r"C:\Python313\python.exe"
    bridge_module: str = "pymt5linux"
    max_restarts: int = 20


@dataclass
class SupervisorState:
    bridge_backoff: int = 2
    restart_counts: Dict[str, int] = field(default_factory=dict)
    processes: Dict[str, object] = field(default_factory=dict)
