from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

from shared.health import check_mt5_bridge, wait_for_bridge
from .config import StrategySpec, SupervisorConfig, SupervisorState


class MT5Supervisor:
    def __init__(self, config: SupervisorConfig, strategies: list[StrategySpec]) -> None:
        self.config = config
        self.strategies = strategies
        self.state = SupervisorState(restart_counts={s.name: 0 for s in strategies})
        self.bridge_proc: Optional[subprocess.Popen] = None

    def _bridge_cmd(self) -> list[str]:
        bridge_python = Path(self.config.wineprefix) / "drive_c" / "Python313" / "python.exe"
        return [
            "wine",
            str(bridge_python),
            "-m",
            self.config.bridge_module,
            self.config.win_python,
            "-p",
            str(self.config.port),
        ]

    def _bridge_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        env["WINEPREFIX"] = self.config.wineprefix
        env.pop("DOTNET_ROOT", None)
        return env

    def start_bridge(self) -> subprocess.Popen:
        self.bridge_proc = subprocess.Popen(
            self._bridge_cmd(),
            env=self._bridge_env(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        return self.bridge_proc

    def bridge_healthy(self) -> bool:
        ok, _ = check_mt5_bridge(self.config.host, self.config.port)
        return ok

    def run(self) -> None:
        while True:
            if not self.bridge_healthy():
                if self.bridge_proc is not None and self.bridge_proc.poll() is None:
                    try:
                        self.bridge_proc.terminate()
                    except Exception:
                        pass
                    time.sleep(2)
                self.start_bridge()
                wait_for_bridge(self.config.host, self.config.port)
                time.sleep(self.state.bridge_backoff)
                self.state.bridge_backoff = min(self.state.bridge_backoff * 2, 60)
                continue

            self.state.bridge_backoff = 2
            time.sleep(5)
