#!/bin/bash
set -euo pipefail

ROOT="/home/chain4655/Documents/Projects/MT5"
LOG_DIR="$ROOT/logs"
SUPERVISOR="$ROOT/supervisors/mt5_bridge_supervisor.py"
PID_FILE="$LOG_DIR/mt5_supervisor.pid"
CLEAN_RESTART="$ROOT/scripts/clean_mt5_restart.py"

mkdir -p "$LOG_DIR"

start() {
    cd "$ROOT"
    if pgrep -f "mt5_bridge_supervisor.py" >/dev/null; then
        echo "Refusing start: mt5_bridge_supervisor.py already running. Use '$0 restart' for clean restart."
        status
        exit 1
    fi
    if lsof -i :18812 >/dev/null 2>&1; then
        echo "Refusing start: port 18812 already occupied. Use '$0 restart' for clean restart."
        lsof -i :18812 || true
        exit 1
    fi
    python3.14 "$SUPERVISOR" > "$LOG_DIR/mt5_supervisor.log" 2>&1 &
    echo $! > "$PID_FILE"
    echo "mt5_bridge_supervisor.py started (PID: $!)"
}

stop() {
    cd "$ROOT"
    python3.14 "$CLEAN_RESTART" stop
    rm -f "$PID_FILE"
    echo "MT5 project stopped"
}

status() {
    cd "$ROOT"
    python3.14 "$CLEAN_RESTART" status
    echo ""
    echo "=== Logs ==="
    tail -n 10 "$LOG_DIR/mt5_supervisor.log" 2>/dev/null || true
}

restart() {
    cd "$ROOT"
    python3.14 "$CLEAN_RESTART" restart | tee "$LOG_DIR/mt5_clean_restart.log"
}

case "${1:-}" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    status)
        status
        ;;
    restart)
        restart
        ;;
    *)
        echo "Usage: $0 {start|stop|status|restart}"
        exit 1
        ;;
esac
