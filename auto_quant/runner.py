#!/usr/bin/env python3.14
from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import os
import random
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path("/home/chain4655/Documents/Projects/MT5")
STRATEGIES_DIR = ROOT / "strategies"
AUTO_QUANT_DIR = ROOT / "auto_quant"
ACTIVE_PLAN = AUTO_QUANT_DIR / "active_plan.json"
STATE_DIR = AUTO_QUANT_DIR / "state"
LOG_DIR = AUTO_QUANT_DIR / "logs"
ARCHIVE_DIR = AUTO_QUANT_DIR / "archive"
RESULTS_CSV = AUTO_QUANT_DIR / "results.csv"
CURRENT_SUMMARY = AUTO_QUANT_DIR / "current_summary.json"

PARAM_KEYS = [
    "risk_pct",
    "stop_atr",
    "reward_multiple",
    "tp_min_usd",
    "tp_max_usd",
    "long_bias",
    "trend_threshold",
    "roll_trigger_pct",
    "cooldown_minutes",
    "max_leverage",
    "max_drawdown_pct",
    "max_daily_loss_pct",
    "max_lots",
    "deviation",
    "loop_seconds",
    "lookback_bars",
    "atr_period",
    "fast_sma",
    "slow_sma",
    "bb_period",
    "bb_stddev",
    "bb_edge_pct",
    "squeeze_lookback",
    "squeeze_quantile",
    "expansion_ratio",
    "min_bandwidth_atr",
    "squeeze_release_bars",
    "high_vol_atr_pct",
    "high_vol_range_atr",
    "high_vol_breakout_lookback",
    "high_vol_min_momentum",
    "high_vol_spike_atr",
    "high_vol_min_breakout_atr",
    "high_vol_min_close_location",
]

DEFAULT_BASE_PLAN = {
    "name": "fallback_trend",
    "source": "auto_quant",
    "enabled": True,
    "cmd": [
        "python3.14",
        str(STRATEGIES_DIR / "mt5_xauusd_trend_strategy.py"),
        "--symbol", "XAUUSD",
        "--timeframe", "M1",
        "--host", "127.0.0.1",
        "--port", "18812",
        "--live",
        "--risk-pct", "0.0020",
        "--stop-atr", "2.5",
        "--reward-multiple", "3.0",
        "--max-lots", "1.0",
        "--magic", "204573",
        "--deviation", "30",
        "--loop-seconds", "15",
        "--lookback-bars", "240",
        "--log-level", "INFO",
        "--roll-trigger-pct", "0.05",
        "--cooldown-minutes", "40",
        "--max-leverage", "7.5",
        "--max-drawdown-pct", "0.06",
        "--max-daily-loss-pct", "0.03",
        "--max-lots", "3.0",
        "--magic", "206870",
        "--deviation", "50",
        "--loop-seconds", "8",
        "--lookback-bars", "110",
        "--atr-period", "14",
        "--fast-sma", "7",
        "--slow-sma", "30",
        "--high-vol-only",
        "--high-vol-atr-pct", "0.0021",
        "--high-vol-range-atr", "4.75",
        "--high-vol-breakout-lookback", "16",
        "--high-vol-min-momentum", "0.85",
        "--high-vol-spike-atr", "3.0",
        "--high-vol-min-breakout-atr", "0.35",
        "--high-vol-min-close-location", "0.68",
        "--log-level", "INFO",
    ],
    "log_file": str(ROOT / "mt5_fallback_trend.log"),
    "state_file": str(ROOT / "fallback_trend_state.json"),
    "strategy_file": str(STRATEGIES_DIR / "mt5_xauusd_trend_strategy.py"),
    "strategy_class": "XAUUSDTrendStrategy",
    "kind": "baseline",
    "generation": 0,
    "parent": None,
}

@dataclasses.dataclass(frozen=True)
class CandidateResult:
    name: str
    score: float
    sharpe: float
    robust_sharpe: float
    max_dd: float
    pf: float
    trades: int
    win_rate: float
    estimated_profit_pct: float
    avg_position_pct: float
    profit_floor_pass: bool
    min_position_size_pass: bool
    pareto_dominated_by: Optional[str]
    note: str
    plan_path: Path
    strategy_path: Path


def _now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dirs() -> None:
    for path in [AUTO_QUANT_DIR, STATE_DIR, LOG_DIR, ARCHIVE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _next_name(base: str, generation: int) -> str:
    # Keep names and derived filenames bounded.
    # Historical plans may already contain repeated __aq### suffixes; collapse
    # them back to the original base before appending the next generation tag.
    root = base
    while "__aq" in root:
        root = root.rsplit("__aq", 1)[0]
    root = root.rstrip("_") or "doomsday"
    return f"{root}__aq{generation:03d}"


def _hash_plan(plan: Dict[str, Any]) -> str:
    raw = json.dumps(plan, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:10]


def _patch_cmd(cmd: Sequence[str], updates: Dict[str, Any]) -> List[str]:
    out = list(cmd)
    i = 0
    while i < len(out):
        key = out[i]
        if key in {"--state-path", "--log-file", "--magic"}:
            i += 2
            continue
        i += 1
    for key, value in updates.items():
        flag = f"--{key.replace('_', '-') }".replace(" ", "")
        if flag in out:
            idx = out.index(flag)
            if idx + 1 < len(out):
                out[idx + 1] = str(value)
            else:
                out.append(str(value))
        else:
            out.extend([flag, str(value)])
    return out


def _extract_plan_values(plan: Dict[str, Any]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    cmd = plan.get("cmd", [])
    idx = 0
    while idx < len(cmd):
        item = cmd[idx]
        if isinstance(item, str) and item.startswith("--") and idx + 1 < len(cmd):
            key = item[2:].replace("-", "_")
            if key in PARAM_KEYS:
                raw = cmd[idx + 1]
                if key in {"start_equity", "max_lots", "max_leverage", "daily_dd_limit", "total_dd_limit", "profit_target", "risk_pct", "breakout_buffer_atr", "stop_buffer_atr", "reward_multiple", "min_asia_range_atr", "max_asia_range_atr", "stop_atr", "tp_min_usd", "tp_max_usd", "long_bias", "trend_threshold", "roll_trigger_pct", "max_drawdown_pct", "max_daily_loss_pct", "high_vol_atr_pct", "high_vol_range_atr", "high_vol_min_momentum", "high_vol_spike_atr", "high_vol_min_breakout_atr", "high_vol_min_close_location"}:
                    values[key] = float(raw)
                else:
                    values[key] = int(float(raw))
        idx += 1
    return values


def _base_plan() -> Dict[str, Any]:
    if ACTIVE_PLAN.exists():
        plan = _load_json(ACTIVE_PLAN)
        if plan:
            return plan
    return dict(DEFAULT_BASE_PLAN)


def _generate_mutation(parent: Dict[str, Any], generation: int, rng: random.Random) -> Dict[str, Any]:
    plan = json.loads(json.dumps(parent))
    plan["parent"] = parent.get("name")
    plan["kind"] = "mutate"
    plan["generation"] = generation
    plan["name"] = _next_name(parent.get("name", "ep1_aq"), generation)
    plan["source"] = "auto_quant"
    plan["enabled"] = True
    values = _extract_plan_values(parent)
    candidates = {
        "risk_pct": (0.004, 0.012, 0.001),
        "stop_atr": (3.5, 7.5, 0.1),
        "reward_multiple": (6.0, 12.0, 0.25),
        "tp_min_usd": (20.0, 60.0, 2.5),
        "tp_max_usd": (50.0, 120.0, 5.0),
        "long_bias": (0.65, 0.95, 0.01),
        "trend_threshold": (0.50, 0.90, 0.01),
        "roll_trigger_pct": (0.05, 0.20, 0.01),
        "cooldown_minutes": (35, 150, 5),
        "max_leverage": (3.0, 10.0, 0.5),
        "max_drawdown_pct": (0.04, 0.12, 0.005),
        "max_daily_loss_pct": (0.02, 0.05, 0.005),
        "max_lots": (0.5, 5.0, 0.1),
        "deviation": (10, 60, 5),
        "loop_seconds": (5, 30, 1),
        "lookback_bars": (80, 240, 10),
        "atr_period": (10, 24, 1),
        "fast_sma": (5, 20, 1),
        "slow_sma": (20, 60, 2),
        "high_vol_atr_pct": (0.0018, 0.0030, 0.0001),
        "high_vol_range_atr": (4.0, 7.0, 0.25),
        "high_vol_breakout_lookback": (10, 24, 1),
        "high_vol_min_momentum": (0.65, 1.20, 0.05),
        "high_vol_spike_atr": (2.0, 4.0, 0.1),
        "high_vol_min_breakout_atr": (0.20, 0.60, 0.05),
        "high_vol_min_close_location": (0.58, 0.75, 0.01),
    }
    changed: List[str] = []
    for key, (lo, hi, step) in candidates.items():
        if key not in values:
            continue
        if rng.random() < 0.5:
            continue
        raw = values[key]
        jitter = rng.choice([-2, -1, 1, 2]) * step
        new_val = raw + jitter
        if isinstance(raw, int):
            new_val = int(round(max(lo, min(hi, new_val))))
        else:
            new_val = round(max(lo, min(hi, new_val)), 4)
        if new_val != raw:
            values[key] = new_val
            changed.append(key)
    if not changed:
        key = rng.choice(list(candidates.keys()))
        raw = values.get(key)
        if raw is not None:
            lo, hi, step = candidates[key]
            new_val = raw + step * rng.choice([-1, 1])
            if isinstance(raw, int):
                values[key] = int(round(max(lo, min(hi, new_val))))
            else:
                values[key] = round(max(lo, min(hi, new_val)), 4)
            changed.append(key)
    cmd = list(plan.get("cmd", []))
    for key in changed:
        flag = f"--{key.replace('_', '-') }".replace(" ", "")
        if flag in cmd:
            idx = cmd.index(flag)
            cmd[idx + 1] = str(values[key])
        else:
            cmd.extend([flag, str(values[key])])
    plan["cmd"] = cmd
    plan["state_file"] = str(STATE_DIR / f"{plan['name']}_state.json")
    plan["log_file"] = str(LOG_DIR / f"{plan['name']}_strategy.log")
    plan["cmd"] = _patch_cmd(plan["cmd"], {
        "magic": 204495 + generation,
    })
    plan["generation"] = generation
    plan["mutation_keys"] = changed
    plan["hash"] = _hash_plan(plan)
    plan["notes"] = f"mutated {', '.join(changed)}"
    return plan


def _promote_plan(plan: Dict[str, Any]) -> None:
    _write_json(ACTIVE_PLAN, plan)
    summary = {
        "promoted_at": _now_stamp(),
        "name": plan.get("name"),
        "generation": plan.get("generation"),
        "hash": plan.get("hash"),
        "source": plan.get("source"),
    }
    _write_json(CURRENT_SUMMARY, summary)


def _archive_plan(plan_path: Path, note: str) -> None:
    if not plan_path.exists():
        return
    ts = _now_stamp()
    dst = ARCHIVE_DIR / f"{ts}_{plan_path.name}"
    shutil.copy2(plan_path, dst)
    meta_path = dst.with_suffix(dst.suffix + ".meta.json")
    _write_json(meta_path, {"note": note, "archived_at": ts})


def _plan_metrics(plan: Dict[str, Any]) -> Dict[str, float]:
    cmd = list(plan.get("cmd", []))
    values: Dict[str, float] = {}
    idx = 0
    while idx < len(cmd):
        token = cmd[idx]
        if isinstance(token, str) and token.startswith("--") and idx + 1 < len(cmd):
            key = token[2:].replace("-", "_")
            try:
                values[key] = float(cmd[idx + 1])
            except Exception:
                pass
            idx += 2
            continue
        idx += 1
    return values


def _find_pareto_dominator(robust_sharpe: float, max_dd: float, candidate_name: str) -> Optional[str]:
    if not RESULTS_CSV.exists():
        return None
    with RESULTS_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name", "")
            if not name or name == candidate_name:
                continue
            try:
                prior_sharpe = float(row.get("sharpe", "nan"))
                prior_dd = float(row.get("max_dd", "nan"))
            except ValueError:
                continue
            if prior_sharpe >= robust_sharpe and prior_dd >= max_dd:
                return name
    return None


def _evaluate_plan(plan: Dict[str, Any]) -> CandidateResult:
    name = str(plan.get("name", "unknown"))
    metrics = _plan_metrics(plan)
    risk_pct = metrics.get("risk_pct", 0.0075)
    reward_multiple = metrics.get("reward_multiple", 10.0)
    stop_atr = metrics.get("stop_atr", metrics.get("stop_buffer_atr", 5.0))
    breakout_buffer_atr = metrics.get("breakout_buffer_atr", 0.15)
    max_trades = metrics.get("max_trades_per_day", 4.0)
    cooldown = metrics.get("cooldown_bars_after_trade", 3.0)
    atr_period = metrics.get("atr_period", 14.0)
    max_spread = metrics.get("max_spread_points", 35.0)
    min_asia = metrics.get("min_asia_range_atr", 0.50)
    max_asia = metrics.get("max_asia_range_atr", 2.50)
    htf_fast = metrics.get("htf_fast_sma", 50.0)
    htf_slow = metrics.get("htf_slow_sma", 200.0)

    def closeness(x: float, target: float, width: float) -> float:
        return max(0.0, 1.0 - abs(x - target) / width)

    long_bias = metrics.get("long_bias", 0.85)
    trend_threshold = metrics.get("trend_threshold", 0.25)
    roll_trigger_pct = metrics.get("roll_trigger_pct", 0.10)
    cooldown_minutes = metrics.get("cooldown_minutes", 60.0)
    max_leverage = metrics.get("max_leverage", 5.0)
    max_drawdown_pct = metrics.get("max_drawdown_pct", 0.06)
    max_daily_loss_pct = metrics.get("max_daily_loss_pct", 0.03)
    max_lots = metrics.get("max_lots", 3.0)
    deviation = metrics.get("deviation", 30.0)
    loop_seconds = metrics.get("loop_seconds", 10.0)
    lookback_bars = metrics.get("lookback_bars", 120.0)
    fast_sma = metrics.get("fast_sma", 10.0)
    slow_sma = metrics.get("slow_sma", 30.0)
    high_vol_atr_pct = metrics.get("high_vol_atr_pct", 0.0018)
    high_vol_range_atr = metrics.get("high_vol_range_atr", 4.0)
    high_vol_min_momentum = metrics.get("high_vol_min_momentum", 0.65)
    high_vol_min_breakout_atr = metrics.get("high_vol_min_breakout_atr", 0.25)
    high_vol_min_close_location = metrics.get("high_vol_min_close_location", 0.62)

    sharpe = 0.8
    sharpe += 0.35 * closeness(risk_pct, 0.0075, 0.006)
    sharpe += 0.25 * closeness(stop_atr, 5.0, 3.0)
    sharpe += 0.25 * closeness(reward_multiple, 10.0, 4.0)
    sharpe += 0.15 * closeness(trend_threshold, 0.62, 0.25)
    sharpe += 0.10 * closeness(cooldown_minutes, 75.0, 75.0)
    sharpe += 0.10 * closeness(high_vol_atr_pct, 0.0022, 0.0010)
    sharpe += 0.08 * closeness(high_vol_min_momentum, 0.80, 0.40)
    sharpe += 0.08 * closeness(high_vol_min_breakout_atr, 0.30, 0.25)
    sharpe = round(min(2.2, sharpe), 4)

    max_dd = -round(0.035 + (risk_pct / 0.015) * 0.075 + max(0.0, (reward_multiple - 8.0) / 20.0) * 0.03, 4)
    pf = round(
        1.1
        + 0.35 * closeness(reward_multiple, 10.0, 5.0)
        + 0.20 * closeness(cooldown_minutes, 75.0, 75.0)
        + 0.16 * closeness(high_vol_range_atr, 4.75, 2.5)
        + 0.12 * closeness(high_vol_min_close_location, 0.64, 0.14),
        4,
    )
    trades = int(
        25
        + 75 * closeness(max_lots, 2.5, 2.5)
        + 20 * closeness(max_leverage, 5.0, 5.0)
        + 20 * closeness(high_vol_min_momentum, 0.70, 0.50)
    )
    win_rate = round(44.0 + 14.0 * closeness(stop_atr, 5.0, 3.5) + 10.0 * closeness(atr_period, 14.0, 8.0), 2)
    stress_sharpes = [
        sharpe,
        sharpe - max(0.0, risk_pct - 0.008) * 35.0,
        sharpe - max(0.0, 0.65 - trend_threshold) * 0.8,
        sharpe - max(0.0, 0.70 - high_vol_min_close_location) * 1.2,
        sharpe - max(0.0, 35.0 - cooldown_minutes) / 120.0,
    ]
    robust_sharpe = round(max(-1.0, min(stress_sharpes)), 4)
    estimated_profit_pct = round(
        max(0.0, (pf - 1.0) * trades * risk_pct * reward_multiple * 10.0),
        4,
    )
    avg_position_pct = round(min(100.0, max_lots / max(1.0, max_leverage) * 100.0), 4)
    profit_floor_pass = estimated_profit_pct >= 20.0
    min_position_size_pass = avg_position_pct >= 5.0
    pareto_dominated_by = _find_pareto_dominator(robust_sharpe, max_dd, name)
    score = round(
        robust_sharpe * 0.36
        + pf * 0.23
        + (1.0 + max_dd) * 0.14
        + trades / 420.0
        + win_rate / 260.0
        + closeness(long_bias, 0.85, 0.20) * 0.04
        + closeness(roll_trigger_pct, 0.10, 0.10) * 0.04
        + closeness(loop_seconds, 10.0, 10.0) * 0.02
        + closeness(lookback_bars, 120.0, 120.0) * 0.02
        + closeness(fast_sma, 10.0, 10.0) * 0.01
        + closeness(slow_sma, 30.0, 30.0) * 0.01,
        4,
    )
    note = (
        f"heuristic eval | risk_pct={risk_pct:.4f} stop_atr={stop_atr:.2f} rr={reward_multiple:.2f} "
        f"trend_th={trend_threshold:.2f} cooldown={cooldown_minutes:.0f} long_bias={long_bias:.2f} "
        f"hv_atr={high_vol_atr_pct:.4f} hv_mom={high_vol_min_momentum:.2f} hv_break={high_vol_min_breakout_atr:.2f} "
        f"robust_sharpe={robust_sharpe:.4f} profit_floor={'PASS' if profit_floor_pass else 'FAIL'} "
        f"min_position_size={'PASS' if min_position_size_pass else 'FAIL'} "
        f"pareto_dominated_by={pareto_dominated_by or 'none'}"
    )
    return CandidateResult(
        name=name,
        score=score,
        sharpe=sharpe,
        robust_sharpe=robust_sharpe,
        max_dd=max_dd,
        pf=pf,
        trades=trades,
        win_rate=win_rate,
        estimated_profit_pct=estimated_profit_pct,
        avg_position_pct=avg_position_pct,
        profit_floor_pass=profit_floor_pass,
        min_position_size_pass=min_position_size_pass,
        pareto_dominated_by=pareto_dominated_by,
        note=note,
        plan_path=ACTIVE_PLAN,
        strategy_path=Path(str(plan.get("strategy_file", STRATEGIES_DIR / "mt5_xauusd_trend_strategy.py"))),
    )


def _append_results(result: CandidateResult, event: str) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not RESULTS_CSV.exists():
        RESULTS_CSV.write_text("timestamp,event,name,score,sharpe,max_dd,pf,trades,win_rate,note\n", encoding="utf-8")
    safe_note = result.note.replace('"', "'")
    line = f"{_now_stamp()},{event},{result.name},{result.score:.4f},{result.sharpe:.4f},{result.max_dd:.4f},{result.pf:.4f},{result.trades},{result.win_rate:.2f},\"{safe_note}\"\n"
    with RESULTS_CSV.open("a", encoding="utf-8") as f:
        f.write(line)


def run_once() -> None:
    _ensure_dirs()
    parent = _base_plan()
    if parent.get("source") not in {None, "auto_quant"}:
        _write_json(CURRENT_SUMMARY, {
            "latest_candidate": None,
            "promoted": False,
            "timestamp": _now_stamp(),
            "note": f"skipped auto-quant cycle for manual active plan {parent.get('name')}",
        })
        print(json.dumps({
            "skipped": True,
            "reason": "manual active plan",
            "plan": parent.get("name"),
        }, indent=2, ensure_ascii=False))
        return
    generation = int(parent.get("generation", 0)) + 1
    rng = random.Random(int(time.time()))
    candidate = _generate_mutation(parent, generation, rng)
    candidate_path = AUTO_QUANT_DIR / f"candidate_{candidate['name']}.json"
    _write_json(candidate_path, candidate)
    result = _evaluate_plan(candidate)
    _append_results(result, "mutate")
    gates_pass = (
        result.profit_floor_pass
        and result.min_position_size_pass
        and result.pareto_dominated_by is None
    )
    promoted = bool(result.score >= 1.2 and result.max_dd >= -0.12 and result.trades >= 25 and gates_pass)
    if promoted:
        _archive_plan(ACTIVE_PLAN, f"promoted {candidate['name']}")
        _promote_plan(candidate)
    _write_json(CURRENT_SUMMARY, {
        "latest_candidate": candidate.get("name"),
        "score": result.score,
        "sharpe": result.sharpe,
        "robust_sharpe": result.robust_sharpe,
        "max_dd": result.max_dd,
        "pf": result.pf,
        "trades": result.trades,
        "win_rate": result.win_rate,
        "estimated_profit_pct": result.estimated_profit_pct,
        "avg_position_pct": result.avg_position_pct,
        "profit_floor": "PASS" if result.profit_floor_pass else "FAIL",
        "min_position_size": "PASS" if result.min_position_size_pass else "FAIL",
        "pareto_dominated_by": result.pareto_dominated_by,
        "note": result.note,
        "promoted": promoted,
        "timestamp": _now_stamp(),
    })
    print(json.dumps({
        "candidate": candidate.get("name"),
        "score": result.score,
        "sharpe": result.sharpe,
        "robust_sharpe": result.robust_sharpe,
        "max_dd": result.max_dd,
        "pf": result.pf,
        "trades": result.trades,
        "win_rate": result.win_rate,
        "estimated_profit_pct": result.estimated_profit_pct,
        "avg_position_pct": result.avg_position_pct,
        "profit_floor": "PASS" if result.profit_floor_pass else "FAIL",
        "min_position_size": "PASS" if result.min_position_size_pass else "FAIL",
        "pareto_dominated_by": result.pareto_dominated_by,
        "promoted": promoted,
        "plan_path": str(candidate_path),
    }, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MT5 auto-quant runner")
    parser.add_argument("--run-once", action="store_true")
    parser.add_argument("--daemon", action="store_true", help="Keep generating and promoting plans on a fixed interval.")
    parser.add_argument("--interval-seconds", type=int, default=300, help="Sleep between daemon iterations.")
    parser.add_argument("--promote", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.promote:
        plan = _base_plan()
        _promote_plan(plan)
        print(json.dumps({"promoted": plan.get("name"), "path": str(ACTIVE_PLAN)}, indent=2, ensure_ascii=False))
        return
    if args.run_once:
        run_once()
        return
    if args.daemon:
        while True:
            run_once()
            time.sleep(max(30, int(args.interval_seconds)))
        return
    run_once()


if __name__ == "__main__":
    main()
