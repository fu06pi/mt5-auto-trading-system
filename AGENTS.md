# AGENTS.md — MT5 Auto Trading System

## Project overview
Python-driven MT5 live-ops/research workspace for XAUUSD strategies.  
GitHub: https://github.com/fu06pi/mt5-auto-trading-system  
Python 3.14, no packaging/manifest files — run scripts directly.  
Venv at `~/.openharness-venv` (not in repo).

## Architecture (layered)
```
MT5 terminal (Wine)
  → pymt5linux bridge (127.0.0.1:18812, secondary bridge 18813)
    → supervisors/mt5_strategy_fleet_supervisor.py
      → reads auto_quant/active_plan.json (gitignored, runtime config)
        → spawns strategy child process (e.g. strategies/mt5_xauusd_trend_strategy.py)
```
Fleet supervisor watches `active_plan.json` content hash and reloads strategy on change.  
Bridge supervisor (`supervisors/mt5_bridge_supervisor.py`) manages Wine-python bridge lifecycle.

## Directory conventions
| Path | Purpose |
|------|---------|
| `strategies/` | Strategy implementations (`mt5_<name>_strategy.py`) |
| `supervisors/` | Bridge + fleet supervisor |
| `shared/` | Base classes, health checks, logging, account metrics |
| `mt5_system/` | Supervisor config dataclasses + base bridge supervisor class |
| `support/` | Pure re-export module — single import point for runtime |
| `auto_quant/` | `active_plan.json`, `active_fleet_plan.json`, `candidate_*.json`, `archive/` (6000 snapshots), `backups/` (142+ pre-edit backups) |
| `tests/` | pytest suite — stubs pymt5linux |
| `scripts/` | Ops helpers (clean restart, log DB, trade audit, round-level monitor) |
| `research/` | Research outputs (CSV, sqlite3 DBs) |
| `spikes/` | Isolated experimental work |
| Root `*.py` | One-off research (`research_*.py`, `backtest_*.py`, `tmp_*.py`) — read-only, never modify `active_plan.json` |

## Key commands
```bash
# Compile check (key runtime files)
python3 -m py_compile strategies/mt5_xauusd_trend_strategy.py supervisors/mt5_strategy_fleet_supervisor.py

# Run all tests
python3 -m pytest tests/ -q

# Single test file
python3 -m pytest tests/test_xauusd_trend_strategy_risk_controls.py -q

# System lifecycle
./run_mt5_system.sh status|start|stop|restart
```

## Test patterns
Tests use a `FakeMT5` / `FakeMetaTrader5` class that stubs `pymt5linux.MetaTrader5`.  
They install it via `sys.modules["pymt5linux"]` then `importlib`-load the strategy module.  
No MT5 terminal or live bridge required. Key test file: `tests/test_xauusd_trend_strategy_risk_controls.py` — covers DD limits, profit target, spread/session filters, loss cooldown, concentration guard, trailing/break-even, false-breakout reversal, chop gating, and more.

## Style
- 4-space indent, `snake_case` for functions/vars/modules, `PascalCase` for classes
- No formatter/linter config — follow surrounding file style
- Strategy files: `mt5_<name>_strategy.py`
- Research scripts: `research_*`, `backtest_*`, `tmp_*` prefixes
- No `__init__.py` imports at root — scripts run as `python3 path/to/script.py`

## Critical gotchas
- **No `pyproject.toml`, `setup.py`, or `requirements.txt`** — all deps (primarily `pymt5linux`) live in the venv
- **No CI/CD** — no `.github/` directory
- **Research scripts must not modify `active_plan.json`** — they are read-only by convention
- **Backup first** before live edits — `auto_quant/backups/` has 142+ timestamped pre-edit snapshots with descriptive names
- **`active_plan.json` and all `auto_quant/*.json` are gitignored** — do not commit runtime state
- Commit style: short imperative, optional conventional prefixes (`feat:`, `chore:`)
- No PR workflow documented — commits are direct to main
- `.hermes/plans/` contains historical agent plans (e.g., Lean integration roadmap) — informational only
- **Account switch to 10k**: See `~/.hermes/skills/trading/trading-system-development-ops/references/account-switch-to-10k-plan.md` — triggered by user saying "我換了更小的帳號"

## Strategy evaluation (current)
- **Keep / develop**: XAUUSD trend main, Meta regime switch, Momentum Surfer, IFVG Sniper, BBRSI ranging
- **Watchlist / research-only**: Bollinger edge squeeze, Doomsday V4, Tide-wave grid, EP1, MNQ confluence
- **Retired**: OCC/Open-Close Cross, XAUUSD confluence pullback, XAUUSD ICT/SMC reversal
