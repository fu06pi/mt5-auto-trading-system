#!/usr/bin/env python3.14
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

HOST = "127.0.0.1"
PORT = 18812
WINEPREFIX = "/home/chain4655/.mt5"
WIN_PY = r"C:\Python313\python.exe"
BRIDGE_MODULE = "pymt5linux"
TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
ROOT = Path("/home/chain4655/Documents/Projects/MT5")
STRATEGIES_DIR = ROOT / "strategies"
AUTO_QUANT_DIR = ROOT / "auto_quant"
LOG = ROOT / "mt5_strategy_fleet_supervisor.log"
BRIDGE_LOG = ROOT / "pymt5linux_bridge.log"
MAX_RESTARTS = 20
BRIDGE_STARTUP_GRACE_SECONDS = 30
BRIDGE_UNHEALTHY_RESTART_THRESHOLD = 3
BRIDGE_SYNC_WAIT_SECONDS = 5
ACTIVE_PLAN = AUTO_QUANT_DIR / "active_plan.json"

DEFAULT_DOOMSDAY_CMD = [
    "python3.14",
    str(STRATEGIES_DIR / "mt5_doomsday_strategy.py"),
    "--symbol",
    "XAUUSD",
    "--timeframe",
    "M5",
    "--host",
    HOST,
    "--port",
    str(PORT),
    "--live",
    "--risk-pct",
    "0.0075",
    "--stop-atr",
    "5.0",
    "--reward-multiple",
    "10.0",
    "--tp-min-usd",
    "30.0",
    "--tp-max-usd",
    "80.0",
    "--long-bias",
    "0.85",
    "--trend-threshold",
    "0.25",
    "--roll-trigger-pct",
    "0.10",
    "--cooldown-minutes",
    "60",
    "--max-leverage",
    "5.0",
    "--max-drawdown-pct",
    "0.06",
    "--max-daily-loss-pct",
    "0.03",
    "--max-lots",
    "3.0",
    "--magic",
    "203493",
    "--deviation",
    "30",
    "--loop-seconds",
    "10",
    "--lookback-bars",
    "120",
    "--atr-period",
    "14",
    "--fast-sma",
    "10",
    "--slow-sma",
    "30",
    "--log-level",
    "INFO",
]

DEFAULT_MOMENTUM_SURFER_CMD = [
    "python3.14",
    str(STRATEGIES_DIR / "mt5_xauusd_momentum_surfer_strategy.py"),
    "--symbol", "XAUUSD",
    "--timeframe", "M5",
    "--host", HOST,
    "--port", str(PORT),
    "--live",
    "--start-equity", "10000.0",
    "--daily-dd-limit", "0.03",
    "--total-dd-limit", "0.06",
    "--profit-target", "0.15",
    "--risk-pct", "0.012",
    "--max-lots", "3.0",
    "--max-leverage", "10.0",
    "--atr-period", "14",
    "--vol-lookback", "50",
    "--mom-lookback", "3",
    "--accel-min", "1.10",
    "--entry-buffer-atr", "0.10",
    "--stop-atr", "2.5",
    "--reward-multiple", "1.0",
    "--trail-trigger-atr", "1.50",
    "--trail-lock-atr", "0.25",
    "--max-spread-points", "150.0",
    "--max-trades-per-day", "9999",
    "--max-consecutive-losses", "5",
    "--cooldown-seconds", "30",
    "--max-hold-minutes", "120",
    "--loop-seconds", "10",
    "--lookback-bars", "120",
    "--state-path", str(AUTO_QUANT_DIR / "state" / "xauusd_momentum_surfer_state.json"),
    "--log-file", str(AUTO_QUANT_DIR / "logs" / "xauusd_momentum_surfer.log"),
    "--deviation", "30",
    "--magic", "210511",
    "--log-level", "INFO",
]

DEFAULT_TUNER_CMD = [
    "python3.14",
    str(AUTO_QUANT_DIR / "runner.py"),
    "--daemon",
    "--interval-seconds",
    "300",
]

DEFAULT_STRATEGY_SPECS = [
    {
        "name": "doomsday",
        "cmd": DEFAULT_DOOMSDAY_CMD,
        "log_file": str(ROOT / "mt5_doomsday_strategy_supervised.log"),
        "restart_on_exit": True,
    },
    {
        "name": "auto_quant_tuner",
        "cmd": DEFAULT_TUNER_CMD,
        "log_file": str(ROOT / "auto_quant" / "runner.log"),
        "source": "auto_quant",
        "restart_on_exit": True,
    },
]
STRATEGY_SPECS = DEFAULT_STRATEGY_SPECS


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    try:
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


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
            if not mt5.initialize(path=TERMINAL_PATH):
                return False
            ti = mt5.terminal_info()
            ai = mt5.account_info()
            return ti is not None and ai is not None
        finally:
            try:
                mt5.shutdown()
            except Exception:
                pass
    except Exception:
        return False


def bridge_server_alive() -> bool:
    try:
        output = subprocess.check_output(["pgrep", "-af", "pymt5linux"], text=True)
        lines = [line for line in output.splitlines() if "pgrep" not in line]
        return bool(lines)
    except subprocess.CalledProcessError:
        return False


def _bridge_pids() -> List[int]:
    try:
        output = subprocess.check_output(["pgrep", "-f", "pymt5linux"], text=True)
    except subprocess.CalledProcessError:
        return []
    pids: List[int] = []
    for raw in output.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            pids.append(int(raw))
        except ValueError:
            continue
    return pids


def _kill_pid(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass


def start_bridge() -> subprocess.Popen:
    env = os.environ.copy()
    env["WINEPREFIX"] = WINEPREFIX
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("DOTNET_ROOT", None)
    cmd = ["wine", f"{WINEPREFIX}/drive_c/Python313/python.exe", "-u", "-m", BRIDGE_MODULE, WIN_PY, "--host", HOST, "-p", str(PORT)]
    log("Starting bridge: " + " ".join(cmd))
    out = open(BRIDGE_LOG, "a", encoding="utf-8")
    return subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, preexec_fn=os.setsid)


def start_strategy(spec: Dict[str, object]) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MT5_SUPERVISED"] = "1"
    if spec.get("source") == "auto_quant":
        env["MT5_AUTO_QUANT"] = "1"
    cmd = spec["cmd"]
    log("Starting strategy[%s]: %s" % (spec["name"], " ".join(cmd)))
    out = open(ROOT / f"{spec['name']}_supervisor.out", "a", encoding="utf-8")
    return subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, preexec_fn=os.setsid)


def _load_active_plan() -> Optional[Dict[str, object]]:
    if not ACTIVE_PLAN.exists():
        return None
    try:
        plan = json.loads(ACTIVE_PLAN.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(plan, dict):
        return None
    strategy_file = str(plan.get("strategy_file", ""))
    strategy_path = Path(strategy_file)
    if strategy_path.suffix != ".py":
        return None
    if strategy_path.parent == Path("."):
        strategy_path = STRATEGIES_DIR / strategy_path
    if strategy_path.parent != STRATEGIES_DIR:
        return None
    allowed_strategy_files = {
        "mt5_doomsday_strategy.py",
        "mt5_doomsday_v4_strategy.py",
        "mt5_xauusd_10d_breakout_strategy.py",
        "mt5_xauusd_momentum_surfer_strategy.py",
        "mt5_xauusd_trend_strategy.py",
    }
    if strategy_path.name not in allowed_strategy_files:
        log(f"Ignoring unsupported active plan: {strategy_path}")
        return None
    cmd = plan.get("cmd")
    if not isinstance(cmd, list) or not cmd:
        return None
    return plan


def _resolve_strategy_specs(plan: Optional[Dict]) -> Tuple[List[Dict], bool]:
    specs: List[Dict] = []
    run_tuner = True
    if plan is None:
        specs.append({
            "name": "doomsday",
            "cmd": list(DEFAULT_DOOMSDAY_CMD),
            "log_file": str(ROOT / "mt5_doomsday_strategy_supervised.log"),
            "restart_on_exit": True,
        })
    else:
        if plan.get("enabled") is False:
            return [], False
        name = str(plan.get("name", "active"))
        cmd = [str(item) for item in plan.get("cmd", [])]
        log_file = str(plan.get("log_file", ""))
        run_tuner = plan.get("source") == "auto_quant"
        specs.append({
            "name": name,
            "cmd": cmd,
            "log_file": log_file or str(ROOT / f"{name}_supervised.log"),
            "restart_on_exit": True,
        })
        for comp in plan.get("complementary", []):
            if comp.get("enabled") is False:
                continue
            cname = str(comp.get("name", "complementary"))
            ccmd = [str(item) for item in comp.get("cmd", [])]
            clog = str(comp.get("log_file", ""))
            if ccmd:
                specs.append({
                    "name": cname,
                    "cmd": ccmd,
                    "log_file": clog or str(ROOT / f"{cname}_supervised.log"),
                    "restart_on_exit": True,
                })
    if run_tuner:
        specs.append({
            "name": "auto_quant_tuner",
            "cmd": DEFAULT_TUNER_CMD,
            "log_file": str(ROOT / "auto_quant" / "runner.log"),
            "source": "auto_quant",
            "restart_on_exit": True,
        })
    return specs, run_tuner


def main() -> None:
    bridge_backoff = 2
    bridge_unhealthy_count = 0
    bridge_started_at = 0.0
    bridge_proc: Optional[subprocess.Popen] = None
    procs: Dict[str, Optional[subprocess.Popen]] = {}
    restart_counts: Dict[str, int] = {}
    completed: Dict[str, bool] = {}
    active_plan_hash: Optional[str] = None
    log("Fleet supervisor started")
    while True:
        try:
            plan = _load_active_plan()
            current_plan_hash = str(plan.get("hash")) if isinstance(plan, dict) and plan.get("hash") else None
            if current_plan_hash != active_plan_hash:
                active_plan_hash = current_plan_hash
                for pname, proc in list(procs.items()):
                    if proc is not None and proc.poll() is None:
                        log(f"Plan changed; terminating [{pname}]")
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                procs.clear()
                restart_counts.clear()
                completed.clear()
                time.sleep(2)

            active_specs, run_tuner = _resolve_strategy_specs(plan)

            bridge_alive = bridge_server_alive()
            healthy = bridge_healthy() if bridge_alive else False
            if healthy:
                bridge_unhealthy_count = 0
                bridge_backoff = 2
            else:
                bridge_unhealthy_count += 1
                if not bridge_alive:
                    if bridge_proc is not None and bridge_proc.poll() is None:
                        log("Bridge process exists but listener absent; waiting for startup")
                    else:
                        now = time.time()
                        if bridge_started_at and now - bridge_started_at < BRIDGE_STARTUP_GRACE_SECONDS:
                            log("Bridge still within startup grace window; waiting")
                        else:
                            for pid in _bridge_pids():
                                log(f"Stopping stale bridge pid {pid}")
                                _kill_pid(pid)
                                time.sleep(0.5)
                            log("Bridge server absent; starting bridge")
                            bridge_proc = start_bridge()
                            bridge_started_at = time.time()
                            bridge_unhealthy_count = 0
                            time.sleep(BRIDGE_SYNC_WAIT_SECONDS)
                else:
                    if bridge_unhealthy_count < BRIDGE_UNHEALTHY_RESTART_THRESHOLD:
                        log("Bridge server exists but is not healthy; waiting for terminal sync")
                    else:
                        log("Bridge health still failing; restarting bridge process")
                        for pid in _bridge_pids():
                            log(f"Stopping unhealthy bridge pid {pid}")
                            _kill_pid(pid)
                        bridge_proc = None
                        bridge_started_at = 0.0
                        bridge_unhealthy_count = 0
                        time.sleep(2)
                        continue
                if not healthy:
                    time.sleep(bridge_backoff)
                    bridge_backoff = min(bridge_backoff * 2, 60)
                    continue

            # Ensure all active specs have tracking entries
            for spec in active_specs:
                sname = spec["name"]
                if sname not in restart_counts:
                    restart_counts[sname] = 0
                if sname not in completed:
                    completed[sname] = False

            # Cleanup stale tracking entries
            active_names = {s["name"] for s in active_specs}
            for sname in list(restart_counts.keys()):
                if sname not in active_names:
                    old_proc = procs.pop(sname, None)
                    if old_proc is not None and old_proc.poll() is None:
                        log(f"Strategy [{sname}] no longer in active plan; terminating")
                        try:
                            old_proc.terminate()
                        except Exception:
                            pass
                    restart_counts.pop(sname, None)
                    completed.pop(sname, None)

            for spec in active_specs:
                name = spec["name"]
                proc = procs.get(name)
                restart_on_exit = bool(spec.get("restart_on_exit", True))
                if completed.get(name):
                    continue
                if proc is None:
                    procs[name] = start_strategy(spec)
                    restart_counts[name] += 1
                    time.sleep(2)
                    continue
                if proc.poll() is not None:
                    if not restart_on_exit:
                        log(f"Strategy[{name}] completed; not restarting")
                        completed[name] = True
                        procs[name] = None
                        continue
                    if restart_counts[name] >= MAX_RESTARTS:
                        log(f"Strategy[{name}] restart limit reached; waiting")
                        continue
                    procs[name] = start_strategy(spec)
                    restart_counts[name] += 1
                    time.sleep(2)
                else:
                    restart_counts[name] = 0

            time.sleep(5)
        except KeyboardInterrupt:
            log("Fleet supervisor interrupted")
            break
        except Exception as e:
            log(f"Fleet supervisor error: {e}")
            time.sleep(bridge_backoff)
            bridge_backoff = min(bridge_backoff * 2, 60)


if __name__ == "__main__":
    main()
