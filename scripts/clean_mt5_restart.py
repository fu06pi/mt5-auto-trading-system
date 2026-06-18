#!/usr/bin/env python3.14
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

ROOT = Path("/home/chain4655/Documents/Projects/MT5")
WINEPREFIX = "/home/chain4655/.mt5"
TERMINAL_EXE = "/home/chain4655/.mt5/drive_c/Program Files/MetaTrader 5/terminal64.exe"
SUPERVISOR = ROOT / "supervisors" / "mt5_bridge_supervisor.py"
PORT = 18812

PATTERNS = [
    "mt5_xauusd_momentum_surfer_strategy.py",
    "mt5_xauusd_asian_reversal_strategy.py",
    "mt5_doomsday_strategy.py",
    "mt5_bridge_supervisor.py",
    "pymt5linux",
    "tmp\\pymt5linux\\server.py",
    "tmp/pymt5linux/server.py",
    "terminal64.exe",
    "wineserver",
]

@dataclass(frozen=True)
class Proc:
    pid: int
    command: str


def run(cmd: List[str], timeout: int = 30, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=check)


def list_processes() -> List[Proc]:
    out = run(["ps", "-eo", "pid=,args="], timeout=10).stdout
    procs: List[Proc] = []
    this_pid = os.getpid()
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid_s, command = line.split(maxsplit=1)
            pid = int(pid_s)
        except ValueError:
            continue
        if pid == this_pid:
            continue
        if "clean_mt5_restart.py" in command:
            continue
        if any(pattern in command for pattern in PATTERNS):
            procs.append(Proc(pid=pid, command=command))
    return procs


def port_open() -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.0)
    try:
        return sock.connect_ex(("127.0.0.1", PORT)) == 0
    finally:
        sock.close()


def port_report() -> str:
    try:
        out = run(["lsof", "-i", f":{PORT}"], timeout=10).stdout.strip()
        return out or f"port {PORT}: free"
    except (subprocess.SubprocessError, FileNotFoundError):
        return f"port {PORT}: {'open' if port_open() else 'free'}"


def kill_processes(procs: Iterable[Proc], sig: signal.Signals) -> None:
    for proc in procs:
        try:
            os.kill(proc.pid, sig)
            print(f"{sig.name} pid={proc.pid} {proc.command[:120]}")
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            print(f"WARN cannot kill pid={proc.pid}: {exc}")


def clean_stop() -> bool:
    print("[1/5] stop duplicate/stale MT5 stack")
    procs = list_processes()
    if procs:
        print(f"found {len(procs)} MT5-related processes")
        kill_processes(procs, signal.SIGTERM)
        time.sleep(4)
    leftovers = list_processes()
    if leftovers:
        print(f"force killing {len(leftovers)} leftovers")
        kill_processes(leftovers, signal.SIGKILL)
        time.sleep(3)
    leftovers = list_processes()
    if leftovers:
        print("ERROR leftovers remain:")
        for proc in leftovers:
            print(f"  pid={proc.pid} {proc.command}")
        return False
    print("all MT5-related processes stopped")

    print("[2/5] verify bridge port release")
    for _ in range(10):
        if not port_open():
            print(f"port {PORT} released")
            return True
        print(f"port {PORT} still open; waiting")
        time.sleep(1)
    print(port_report())
    return not port_open()


def start_terminal() -> subprocess.Popen[bytes]:
    print("[3/5] start MT5 terminal")
    env = os.environ.copy()
    env["WINEPREFIX"] = WINEPREFIX
    proc = subprocess.Popen(
        ["wine", TERMINAL_EXE, "/portable"],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(f"terminal launcher pid={proc.pid}")
    time.sleep(20)
    terminal_procs = [p for p in list_processes() if "terminal64.exe" in p.command]
    print(f"terminal process count={len(terminal_procs)}")
    for p in terminal_procs:
        print(f"  pid={p.pid} {p.command[:140]}")
    return proc


def probe_mt5_initialize() -> bool:
    print("[4/5] probe Wine MetaTrader5.initialize() before bridge")
    code = r'''
import MetaTrader5 as mt5
print('probe:start', flush=True)
ok = mt5.initialize(timeout=20000)
print('probe:initialize', ok, 'err', mt5.last_error(), flush=True)
if ok:
    ai = mt5.account_info()
    ti = mt5.terminal_info()
    print('probe:account', getattr(ai, 'login', None), getattr(ai, 'server', None), flush=True)
    print('probe:terminal_trade_allowed', getattr(ti, 'trade_allowed', None), flush=True)
    mt5.shutdown()
'''
    try:
        res = run(["timeout", "45", "wine", f"{WINEPREFIX}/drive_c/Python313/python.exe", "-c", code], timeout=60)
    except subprocess.TimeoutExpired:
        print("probe timeout")
        return False
    print(res.stdout.strip())
    if res.stderr.strip():
        print(res.stderr.strip()[-1200:])
    return "probe:initialize True" in res.stdout


def start_supervisor() -> subprocess.Popen[bytes]:
    print("[5/5] start bridge supervisor")
    proc = subprocess.Popen(
        ["python3.14", str(SUPERVISOR)],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(f"supervisor pid={proc.pid}")
    time.sleep(15)
    print(port_report())
    return proc


def status() -> None:
    print("=== processes ===")
    procs = list_processes()
    if not procs:
        print("none")
    for proc in procs:
        print(f"pid={proc.pid} {proc.command}")
    print("=== port ===")
    print(port_report())


def main() -> int:
    action = sys.argv[1] if len(sys.argv) > 1 else "restart"
    if action == "status":
        status()
        return 0
    if action == "stop":
        return 0 if clean_stop() else 1
    if action != "restart":
        print("Usage: clean_mt5_restart.py {restart|stop|status}")
        return 2

    if not clean_stop():
        return 1
    start_terminal()
    if not probe_mt5_initialize():
        print("BLOCKED: MT5 Python IPC initialize failed; supervisor not started to avoid restart loop.")
        status()
        return 1
    start_supervisor()
    status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
