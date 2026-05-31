#!/usr/bin/env python3.14
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HOST = "127.0.0.1"
PORT = 18812
WINEPREFIX = "/home/chain4655/.mt5"
WIN_PY = r"Z:\\home\\chain4655\\.mt5\\drive_c\\Python313\\python.exe"
BRIDGE_MODULE = "pymt5linux"
ROOT = Path("/home/chain4655/Documents/Projects/MT5")
STRATEGIES_DIR = ROOT / "strategies"
AUTO_QUANT_DIR = ROOT / "auto_quant"
AUTO_QUANT_MANIFEST = AUTO_QUANT_DIR / "active_plan.json"
BASE_STRATEGY_SCRIPT = STRATEGIES_DIR / "mt5_doomsday_strategy.py"
DEFAULT_STRATEGY_CMD = [
    "python3.14",
    str(BASE_STRATEGY_SCRIPT),
    "--symbol",
    "XAUUSD",
    "--timeframe",
    "M5",
    "--host",
    HOST,
    "--port",
    str(PORT),
    "--live",
]
DEFAULT_STRATEGY_LOG = ROOT / "mt5_doomsday_strategy_supervised.log"
LOG = Path("/home/chain4655/Documents/Projects/MT5/mt5_bridge_supervisor.log")
MAX_STRATEGY_RESTARTS = 20


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    try:
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError as exc:
        log(f"Log write failed: {exc}")


def port_open() -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.5)
    try:
        return s.connect_ex((HOST, PORT)) == 0
    finally:
        s.close()


def bridge_healthy() -> bool:
    if not port_open():
        return False
    try:
        from pymt5linux import MetaTrader5
        mt5 = MetaTrader5(host=HOST, port=PORT)
        try:
            if not mt5.initialize():
                return False
            # initialize() succeeding is the practical health gate here.
            # terminal_info/account_info can time out on this Wine build even
            # when the bridge can serve trades and rates.
            return True
        finally:
            try:
                mt5.shutdown()
            except Exception as exc:
                log(f"Bridge shutdown failed: {exc}")
    except Exception:
        return False


def start_bridge() -> subprocess.Popen:
    env = os.environ.copy()
    env["WINEPREFIX"] = WINEPREFIX
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("DOTNET_ROOT", None)
    cmd = [
        "wine",
        f"{WINEPREFIX}/drive_c/Python313/python.exe",
        "-u",
        "-m",
        BRIDGE_MODULE,
        WIN_PY,
        "--host",
        HOST,
        "-p",
        str(PORT),
    ]
    log("Starting bridge: " + " ".join(cmd))
    log_path = ROOT / "pymt5linux_bridge.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out = open(log_path, "a", encoding="utf-8")
    out.write(f"\n=== bridge start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    out.flush()
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=out,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def _default_strategy_spec() -> Dict[str, object]:
    return {
        "name": "ep1",
        "cmd": DEFAULT_STRATEGY_CMD,
        "log_file": str(DEFAULT_STRATEGY_LOG),
        "state_file": str(ROOT / "xauusd_ep1_state.json"),
        "enabled": True,
        "source": "default",
    }


def _ensure_auto_quant_dirs() -> None:
    for path in [AUTO_QUANT_DIR, AUTO_QUANT_DIR / "logs", AUTO_QUANT_DIR / "state"]:
        path.mkdir(parents=True, exist_ok=True)


def _load_manifests() -> List[Dict[str, object]]:
    _ensure_auto_quant_dirs()
    if not AUTO_QUANT_MANIFEST.exists():
        return [_default_strategy_spec()]
    try:
        data = json.loads(AUTO_QUANT_MANIFEST.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return [_default_strategy_spec()]
        base = _default_strategy_spec()
        base.update({k: v for k, v in data.items() if k in {"name", "cmd", "log_file", "state_file", "enabled", "source"}})
        specs = [base]
        for comp in data.get("complementary", []):
            if isinstance(comp, dict):
                c_enabled = bool(comp.get("enabled", True))
                c_cmd = list(comp.get("cmd") or [])
                if c_enabled and c_cmd:
                    specs.append({
                        "name": str(comp.get("name", "complementary")),
                        "cmd": c_cmd,
                        "log_file": str(comp.get("log_file", str(DEFAULT_STRATEGY_LOG))),
                        "state_file": str(comp.get("state_file", "")),
                        "enabled": True,
                        "source": "auto_quant",
                    })
        return specs
    except Exception as exc:
        log(f"Failed to load auto-quant manifest, using default strategy: {exc}")
        return [_default_strategy_spec()]


def _strategy_log_path(spec: Dict[str, object]) -> Path:
    return Path(str(spec.get("log_file") or DEFAULT_STRATEGY_LOG))


def _start_strategy(spec: Dict[str, object]) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MT5_SUPERVISED"] = "1"
    if spec.get("source") == "auto_quant":
        env["MT5_AUTO_QUANT"] = "1"
    cmd = list(spec.get("cmd") or DEFAULT_STRATEGY_CMD)
    log_file = _strategy_log_path(spec)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    state_file = Path(str(spec.get("state_file") or (ROOT / "xauusd_ep1_state.json")))
    state_file.parent.mkdir(parents=True, exist_ok=True)
    log("Starting strategy[%s]: %s" % (spec.get("name", "unknown"), " ".join(cmd)))
    out = open(log_file, "a", encoding="utf-8")
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=out,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


BACKOFF_CAP = 300


def main() -> None:
    bridge_backoff = 2
    bridge_proc: Optional[subprocess.Popen] = None
    strategy_procs: Dict[str, Optional[subprocess.Popen]] = {}
    strategy_restart_counts: Dict[str, int] = {}
    strategy_backoff_untils: Dict[str, float] = {}
    strategy_backoff_seconds: Dict[str, int] = {}
    last_manifest_signature: Optional[str] = None
    log("Supervisor started")
    while True:
        try:
            specs = _load_manifests()
            manifest_signature = json.dumps(specs, sort_keys=True, ensure_ascii=False)
            if manifest_signature != last_manifest_signature:
                last_manifest_signature = manifest_signature
                for sname, sproc in list(strategy_procs.items()):
                    if sproc is not None and sproc.poll() is None:
                        log(f"Manifest changed; terminating [{sname}]")
                        try:
                            sproc.terminate()
                        except Exception:
                            pass
                strategy_procs.clear()
                strategy_restart_counts.clear()
                strategy_backoff_untils.clear()
                strategy_backoff_seconds.clear()
                time.sleep(2)

            bridge_ok = bridge_healthy()
            if not bridge_ok:
                if bridge_proc is not None and bridge_proc.poll() is None:
                    log("Bridge process alive but unhealthy; restarting")
                    try:
                        bridge_proc.terminate()
                    except Exception:
                        pass
                    time.sleep(2)
                bridge_proc = start_bridge()
                time.sleep(8)
                bridge_ok = bridge_healthy()
                if bridge_ok:
                    log("Bridge recovered")
                    bridge_backoff = 2
                else:
                    log(f"Bridge not healthy after start; retrying in {bridge_backoff}s")
                    time.sleep(bridge_backoff)
                    bridge_backoff = min(bridge_backoff * 2, 60)
                    continue
            else:
                bridge_backoff = 2

            now = time.time()
            for spec in specs:
                sname = str(spec.get("name", "unknown"))
                if sname not in strategy_restart_counts:
                    strategy_restart_counts[sname] = 0
                if sname not in strategy_backoff_untils:
                    strategy_backoff_untils[sname] = 0.0
                if sname not in strategy_backoff_seconds:
                    strategy_backoff_seconds[sname] = 5

                if not bool(spec.get("enabled", True)):
                    sproc = strategy_procs.get(sname)
                    if sproc is not None and sproc.poll() is None:
                        log(f"Strategy [{sname}] paused by manifest")
                        try:
                            sproc.terminate()
                        except Exception:
                            pass
                    strategy_procs[sname] = None
                    continue

                if now < strategy_backoff_untils.get(sname, 0.0):
                    continue

                sproc = strategy_procs.get(sname)
                if sproc is None or sproc.poll() is not None:
                    rc = sproc.poll() if sproc is not None else None
                    risk_stop = rc is not None and rc != 0
                    if strategy_restart_counts.get(sname, 0) >= MAX_STRATEGY_RESTARTS:
                        if risk_stop:
                            log(f"Strategy [{sname}] restart limit (risk shutdown); backoff 5min")
                            strategy_backoff_untils[sname] = time.time() + 300
                            strategy_restart_counts[sname] = max(0, strategy_restart_counts[sname] - 1)
                            continue
                        log(f"Strategy [{sname}] restart limit reached")
                        continue

                    if risk_stop:
                        bs = min(int(strategy_backoff_seconds.get(sname, 5) * 2), BACKOFF_CAP)
                        strategy_backoff_seconds[sname] = bs
                        strategy_backoff_untils[sname] = time.time() + bs
                        log(f"Strategy [{sname}] risk shutdown (rc={rc}); backoff {bs}s")
                        strategy_restart_counts[sname] += 1
                        continue

                    strategy_procs[sname] = _start_strategy(spec)
                    strategy_restart_counts[sname] += 1
                    strategy_backoff_seconds[sname] = 5
                else:
                    strategy_backoff_seconds[sname] = 5
                    strategy_restart_counts[sname] = 0
                    strategy_backoff_untils[sname] = 0.0

            # Clean stale entries
            active_names = {s.get("name", "unknown") for s in specs}
            for sname in list(strategy_procs.keys()):
                if sname not in active_names:
                    sproc = strategy_procs.pop(sname, None)
                    if sproc is not None and sproc.poll() is None:
                        log(f"Strategy [{sname}] removed; terminating")
                        try:
                            sproc.terminate()
                        except Exception:
                            pass
                    strategy_restart_counts.pop(sname, None)
                    strategy_backoff_untils.pop(sname, None)
                    strategy_backoff_seconds.pop(sname, None)

            time.sleep(5)
        except KeyboardInterrupt:
            log("Supervisor interrupted")
            break
        except Exception as e:
            log(f"Supervisor error: {e}")
            time.sleep(bridge_backoff)
            bridge_backoff = min(bridge_backoff * 2, 60)


if __name__ == "__main__":
    main()
