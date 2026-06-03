# MT5 Auto Trading System

Python-driven MetaTrader 5 research and live-ops workspace for XAUUSD strategy development, backtesting, risk controls, and supervised execution.

> Status: research + demo/live operations tooling. Treat every strategy as experimental unless it has passed backtest, paper/demo, and operational review gates.

## What this repo contains

- **Live strategy code** under `strategies/`
  - Main current runtime: `strategies/mt5_xauusd_trend_strategy.py`
  - Other research/runtime candidates: Momentum Surfer, Doomsday, Tide Wave Grid, regime switch, Bollinger squeeze, etc.
- **Supervisor / bridge operations** under `supervisors/` and `scripts/`
  - `supervisors/mt5_strategy_fleet_supervisor.py` watches `auto_quant/active_plan.json` and runs the selected strategy child process.
  - `supervisors/mt5_bridge_supervisor.py` and `scripts/clean_mt5_restart.py` help manage the MT5 / Wine / pymt5linux bridge layer.
- **Research scripts** at repo root
  - Weekly stability checks, complement strategy studies, session-window backtests, and strategy comparison scripts.
- **Shared helpers** under `shared/`, `support/`, and `mt5_system/`
  - Account metrics, health checks, base runtime abstractions, logging helpers.
- **Tests** under `tests/`
  - Current key safety suite: `tests/test_xauusd_trend_strategy_risk_controls.py`.

## Architecture overview

```text
MT5 terminal / Wine
        ↓
pymt5linux bridge 127.0.0.1:18812
        ↓
supervisors/mt5_strategy_fleet_supervisor.py
        ↓
auto_quant/active_plan.json  (runtime plan; ignored by git)
        ↓
strategies/mt5_xauusd_trend_strategy.py
        ↓
auto_quant/state/*.json + auto_quant/logs/*.log  (runtime state/logs; ignored by git)
```

The live system is intentionally split into layers:

1. **Terminal / bridge** — MT5 GUI login, AutoTrading, and local IPC bridge readiness.
2. **Fleet supervisor** — reads `active_plan.json`, restarts the strategy child only when the plan generation changes.
3. **Strategy child** — executes signal logic, risk checks, order sizing, trailing/exit logic, and structured logs.
4. **State / backups** — runtime state and pre-edit backups stay local and are excluded from GitHub.
5. **Research layer** — deterministic scripts generate CSV / Markdown reports without touching live plans.

## Current main strategy direction

The current XAUUSD codebase has evolved from a trend-following sleeve into a portfolio-sleeve research framework:

- **Trend-following sleeve**
  - SMA / HTF trend filters
  - ATR-based stops and rewards
  - startup warmup, spread filter, max hold, cooldowns
- **False-breakout complement sleeve**
  - SELL-only upthrust reversal in bearish HTF conditions
  - Configured via flags such as `--enable-false-breakout-reversal` and `--false-breakout-direction SELL_ONLY`
  - Intended as a low-risk complement / independent sleeve, not a generic overlay
- **Portfolio sleeve model**
  - Research supports standalone trend, standalone complement, simple overlay, and parallel independent sleeves.
  - Current research conclusion: avoid simple overlay; if combined, use independent processes / magic numbers with shared exposure and DD caps.

## Important runtime files not committed

These files are intentionally ignored:

- `auto_quant/active_plan.json`
- `auto_quant/state/`
- `auto_quant/logs/`
- `auto_quant/backups/`
- `auto_quant/test_plans/`
- `backtest_reports*/`
- generated `.csv`, `.log`, `.out`, images, and screenshots
- credentials / tokens / MT5 account files

This prevents accidental publication of account state, logs, reports, local runtime snapshots, or secrets.

## Quick commands

### Check repo status

```bash
git status --short --branch
```

### Compile key Python files

```bash
python3 -m py_compile \
  strategies/mt5_xauusd_trend_strategy.py \
  supervisors/mt5_strategy_fleet_supervisor.py
```

### Run current safety tests

```bash
python3 -m pytest tests/test_xauusd_trend_strategy_risk_controls.py -q
```

### Check MT5 local runtime wrapper

```bash
./run_mt5_system.sh status
```

### Start / stop / restart MT5 bridge supervisor wrapper

```bash
./run_mt5_system.sh start
./run_mt5_system.sh stop
./run_mt5_system.sh restart
```

Use these with care. For live account operations, verify MT5 login, AutoTrading, bridge readiness, account equity, open positions, pending orders, and strategy child process separately.

## Research scripts

Recent research additions:

- `research_xauusd_weekly_complement_backtest.py`
  - Weekly comparison for trend-only, complement-only, simple overlay, and parallel sleeve models.
- `research_xauusd_trend_plus_complement_backtest.py`
  - Main trend + false-breakout complement simulator.
- `research_xauusd_trend_plus_complement_long_cache_backtest.py`
  - Longer-cache / range-aware complement study.
- `research_xauusd_complement_stability.py`
  - Parameter / segment / stability diagnostics for complement strategy.
- `research_xauusd_inverse_left_bottom_backtest.py`
  - Opposite-side / left-side bottom-fishing research.
- `research_xauusd_london_ny_session_backtest.py`
  - Read-only session-window experiment; does not modify live `active_plan.json`.
- `spikes/001-false-breakout-reversal/`
  - Early spike for upthrust / spring reversal logic.

Generated reports are excluded from git. Re-run scripts locally to regenerate CSV / Markdown outputs.

## Safety rules

- Do **not** commit secrets, MT5 account files, logs, state snapshots, or generated reports.
- Before live parameter edits, back up `auto_quant/active_plan.json` and state files.
- Prefer editing `active_plan.json` and letting the fleet supervisor reload the strategy child; avoid opening duplicate MT5 terminals or duplicate strategy processes.
- After any live restart, verify:
  - MT5 terminal is logged in and AutoTrading is enabled.
  - pymt5linux bridge responds on `127.0.0.1:18812`.
  - `active_plan.json` is enabled and the intended strategy child command is running.
  - account equity, positions, orders, and strategy-owned magic numbers match expectations.
  - latest strategy log shows current bar / risk loop, not stale output.
- Research-only scripts must not modify live `active_plan.json`.

## Development timeline

- **2026-04** — MQL5 EA inventory and Pine Script porting experiments.
- **2026-05 early** — Python live strategy + MT5 bridge + supervisor workflow formed.
- **2026-05 mid** — Prop / FTMO-style risk gates added: daily DD, total DD, profit target, warmup, cooldowns.
- **2026-05 late** — Live parameter governance: backups, half-close, trailing stability, per-order lot caps, concentration guard, account-switch recovery.
- **2026-06** — Complement sleeve research, false-breakout reversal code, weekly stability reports, and low-risk complement runtime plan.

## Repository

GitHub: https://github.com/fu06pi/mt5-auto-trading-system
