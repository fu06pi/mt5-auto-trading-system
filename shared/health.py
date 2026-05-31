from __future__ import annotations

import logging
import socket
import subprocess
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def check_port(host: str, port: int, timeout: float = 1.5) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        return sock.connect_ex((host, port)) == 0
    finally:
        sock.close()


def check_ttb_process() -> bool:
    try:
        result = subprocess.run(["pgrep", "-f", "TTB.exe"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def check_mt5_bridge(
    host: str = "127.0.0.1",
    port: int = 18812,
    timeout: float = 3.0,
) -> Tuple[bool, Optional[str]]:
    if not check_port(host, port, timeout):
        return False, "port closed"

    try:
        from pymt5linux import MetaTrader5

        mt5 = MetaTrader5(host=host, port=port)
        try:
            if not mt5.initialize():
                return False, "init failed"
            ti = mt5.terminal_info()
            ai = mt5.account_info()
            if ti is None or ai is None:
                return False, "no terminal/account"
            return True, "ok"
        except Exception as e:
            return False, str(e)
        finally:
            try:
                mt5.shutdown()
            except Exception:
                pass
    except ImportError as e:
        return False, f"import error: {e}"
    except Exception as e:
        return False, str(e)


def wait_for_bridge(
    host: str = "127.0.0.1",
    port: int = 18812,
    max_attempts: int = 10,
    backoff_base: float = 2.0,
) -> bool:
    for attempt in range(max_attempts):
        ok, msg = check_mt5_bridge(host, port)
        if ok:
            logger.info("Bridge ready after %s attempts", attempt + 1)
            return True

        logger.warning("Bridge not ready (attempt %s/%s): %s", attempt + 1, max_attempts, msg)
        if attempt < max_attempts - 1:
            time.sleep(min(backoff_base**attempt, 30.0))

    return False
