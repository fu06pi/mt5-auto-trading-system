#!/usr/bin/env python3
"""Classify MT5 closed trades by market regime and summarize performance.

Research-only: reads exported closed-position CSVs and cached XAUUSD daily bars;
does not touch active_plan.json, processes, or MT5.
"""
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

TRADE_CSV = Path(
    "/home/chain4655/Documents/backtest_reports/mt5_all_accounts_consolidated_20260619/"
    "all_authoritative_closed_positions.csv"
)
DAILY_CSV = Path(
    "/home/chain4655/Documents/backtest_reports/xauusd_multitf_klines/mt5/"
    "XAUUSD_D1_mt5_5666bars.csv"
)
OUT_DIR = Path("/home/chain4655/Documents/backtest_reports/mt5_market_regime_filter_20260619")
ACTIVE_PLAN = Path("/home/chain4655/Documents/Projects/MT5/auto_quant/active_plan.json")

# Manual seed calendar for high-impact USD events that intersect the available
# realized-trade window. Extend this dict as more history is imported.
# Dates are event dates in the XAUUSD/MT5 trade date basis.
MAJOR_NEWS_DATES: Dict[str, List[str]] = {
    "2026-05-01": ["NFP"],
    "2026-05-13": ["CPI"],
    "2026-05-20": ["FOMC_MINUTES"],
    "2026-06-05": ["NFP"],
    "2026-06-11": ["CPI"],
    "2026-06-17": ["FOMC"],
}

ATR_PERIOD = 14
ATR_AVG_DAYS = 20
HIGH_VOL_MULT = 1.5
LOW_VOL_MULT = 0.75
NOISE_ABS_NET = 4.0


@dataclasses.dataclass(frozen=True)
class DailyRegime:
    date: str
    open: float
    high: float
    low: float
    close: float
    atr14: float
    atr20_avg: float
    atr_ratio: float
    body_range_ratio: float
    regime: str
    regime_label: str
    news_tags: str


@dataclasses.dataclass(frozen=True)
class TradeRow:
    account_login: str
    position_id: str
    open_time: str
    close_time: str
    symbol: str
    side: str
    net: float
    magic_open: str
    strategy_bucket: str
    noise_abs_lt4: bool
    source_artifact: str


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def load_daily_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rows.append(
                {
                    "date": raw["time"][:10],
                    "open": parse_float(raw["open"]),
                    "high": parse_float(raw["high"]),
                    "low": parse_float(raw["low"]),
                    "close": parse_float(raw["close"]),
                }
            )
    return rows


def true_range(row: Dict[str, Any], prev_close: Optional[float]) -> float:
    high = float(row["high"])
    low = float(row["low"])
    if prev_close is None:
        return high - low
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def moving_average(values: Sequence[float], end_index: int, length: int) -> Optional[float]:
    start = end_index - length + 1
    if start < 0:
        return None
    window = values[start : end_index + 1]
    if len(window) != length:
        return None
    return sum(window) / length


def build_daily_regimes(rows: Sequence[Dict[str, Any]]) -> Dict[str, DailyRegime]:
    trs: List[float] = []
    prev_close: Optional[float] = None
    for row in rows:
        trs.append(true_range(row, prev_close))
        prev_close = float(row["close"])

    atr14_values: List[Optional[float]] = []
    for index in range(len(rows)):
        atr14_values.append(moving_average(trs, index, ATR_PERIOD))

    out: Dict[str, DailyRegime] = {}
    for index, row in enumerate(rows):
        atr14 = atr14_values[index]
        if atr14 is None:
            continue
        prior_atrs = [value for value in atr14_values[max(0, index - ATR_AVG_DAYS) : index] if value]
        if len(prior_atrs) < min(ATR_AVG_DAYS, 10):
            continue
        atr20_avg = sum(prior_atrs) / len(prior_atrs)
        atr_ratio = atr14 / atr20_avg if atr20_avg > 0 else 0.0
        day_range = max(float(row["high"]) - float(row["low"]), 1e-9)
        body_range_ratio = abs(float(row["close"]) - float(row["open"])) / day_range
        date = str(row["date"])
        news_tags = MAJOR_NEWS_DATES.get(date, [])

        if news_tags:
            regime = "A"
            label = "major_news_day"
        elif atr_ratio >= HIGH_VOL_MULT:
            regime = "B"
            label = "high_vol_trend_day"
        elif atr_ratio <= LOW_VOL_MULT and body_range_ratio <= 0.35:
            regime = "D"
            label = "low_vol_chop_day"
        else:
            regime = "C"
            label = "normal_day"

        out[date] = DailyRegime(
            date=date,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            atr14=round(atr14, 5),
            atr20_avg=round(atr20_avg, 5),
            atr_ratio=round(atr_ratio, 5),
            body_range_ratio=round(body_range_ratio, 5),
            regime=regime,
            regime_label=label,
            news_tags=";".join(news_tags),
        )
    return out


def load_trades(path: Path) -> List[TradeRow]:
    trades: List[TradeRow] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            net = parse_float(raw.get("net", ""), parse_float(raw.get("profit", "")))
            trades.append(
                TradeRow(
                    account_login=str(raw.get("account_login", "")),
                    position_id=str(raw.get("position_id", "")),
                    open_time=str(raw.get("open_time", "")),
                    close_time=str(raw.get("close_time", "")),
                    symbol=str(raw.get("symbol", "")),
                    side=str(raw.get("side", "")),
                    net=net,
                    magic_open=str(raw.get("magic_open", "")),
                    strategy_bucket=str(raw.get("strategy_bucket", "")),
                    noise_abs_lt4=parse_bool(raw.get("noise_abs_lt4", "")) or abs(net) < NOISE_ABS_NET,
                    source_artifact=str(raw.get("source_artifact", "")),
                )
            )
    return trades


def metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    clean_rows = [row for row in rows if not row["noise_abs_lt4"]]
    wins = [row for row in clean_rows if row["net"] > 0]
    losses = [row for row in clean_rows if row["net"] < 0]
    gross_profit = sum(float(row["net"]) for row in wins)
    gross_loss = -sum(float(row["net"]) for row in losses)
    curve = 0.0
    peak = 0.0
    max_dd = 0.0
    for row in sorted(clean_rows, key=lambda item: item["close_time"]):
        curve += float(row["net"])
        peak = max(peak, curve)
        max_dd = min(max_dd, curve - peak)
    return {
        "trades_raw": len(rows),
        "trades_clean_abs4": len(clean_rows),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(100.0 * len(wins) / len(clean_rows), 2) if clean_rows else 0.0,
        "net": round(sum(float(row["net"]) for row in clean_rows), 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "pf": round(gross_profit / gross_loss, 3) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0),
        "avg_net": round(sum(float(row["net"]) for row in clean_rows) / len(clean_rows), 2) if clean_rows else 0.0,
        "closed_curve_max_dd": round(max_dd, 2),
        "first_close": min((row["close_time"] for row in clean_rows), default=""),
        "last_close": max((row["close_time"] for row in clean_rows), default=""),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def active_plan_hash() -> Optional[str]:
    if not ACTIVE_PLAN.exists():
        return None
    import hashlib

    return hashlib.sha256(ACTIVE_PLAN.read_bytes()).hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plan_before = active_plan_hash()
    regimes = build_daily_regimes(load_daily_rows(DAILY_CSV))
    trades = load_trades(TRADE_CSV)

    regime_dates = sorted(regimes.keys())

    def regime_for_trade_date(date_text: str) -> tuple[Optional[DailyRegime], str]:
        if date_text in regimes:
            return regimes[date_text], date_text
        # Weekend / broker-history timestamps can close on Saturday after the Friday
        # session. Map them to the latest available daily trading bar instead of
        # losing the regime label.
        prior_dates = [candidate for candidate in regime_dates if candidate <= date_text]
        if prior_dates:
            basis = prior_dates[-1]
            return regimes[basis], basis
        return None, ""

    enriched: List[Dict[str, Any]] = []
    for trade in trades:
        trade_date = trade.close_time[:10] or trade.open_time[:10]
        regime, basis_date = regime_for_trade_date(trade_date)
        if regime is None:
            regime_values = {
                "date": "",
                "regime": "UNKNOWN",
                "regime_label": "missing_daily_bar",
                "news_tags": "",
                "atr14": "",
                "atr20_avg": "",
                "atr_ratio": "",
                "body_range_ratio": "",
            }
        else:
            regime_values = dataclasses.asdict(regime)
        enriched.append(
            {
                **dataclasses.asdict(trade),
                "trade_date": trade_date,
                "regime_basis_date": basis_date,
                **regime_values,
            }
        )

    by_regime: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_regime_strategy: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    by_regime_symbol: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        by_regime[str(row["regime"])].append(row)
        by_regime_strategy[(str(row["regime"]), str(row["strategy_bucket"]))].append(row)
        by_regime_symbol[(str(row["regime"]), str(row["symbol"]))].append(row)

    regime_rows = []
    for regime_key, rows in sorted(by_regime.items()):
        label = rows[0].get("regime_label", "") if rows else ""
        regime_rows.append({"regime": regime_key, "regime_label": label, **metrics(rows)})

    strategy_rows = []
    for (regime_key, strategy), rows in sorted(by_regime_strategy.items()):
        label = rows[0].get("regime_label", "") if rows else ""
        strategy_rows.append(
            {"regime": regime_key, "regime_label": label, "strategy_bucket": strategy, **metrics(rows)}
        )

    symbol_rows = []
    for (regime_key, symbol), rows in sorted(by_regime_symbol.items()):
        label = rows[0].get("regime_label", "") if rows else ""
        symbol_rows.append({"regime": regime_key, "regime_label": label, "symbol": symbol, **metrics(rows)})

    daily_rows = [dataclasses.asdict(row) for row in regimes.values()]
    write_csv(OUT_DIR / "trades_enriched_by_market_regime.csv", enriched)
    write_csv(OUT_DIR / "summary_by_regime_clean_abs4.csv", regime_rows)
    write_csv(OUT_DIR / "summary_by_regime_strategy_clean_abs4.csv", strategy_rows)
    write_csv(OUT_DIR / "summary_by_regime_symbol_clean_abs4.csv", symbol_rows)
    write_csv(OUT_DIR / "daily_market_regimes.csv", daily_rows)

    diagnostics = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "research_only": True,
        "trade_csv": str(TRADE_CSV),
        "daily_csv": str(DAILY_CSV),
        "out_dir": str(OUT_DIR),
        "trade_rows": len(trades),
        "daily_regime_rows": len(regimes),
        "major_news_dates": MAJOR_NEWS_DATES,
        "rules": {
            "A": "major USD news date: CPI/NFP/FOMC/FOMC minutes manual seed",
            "B": f"ATR14 > prior {ATR_AVG_DAYS} ATR14 average * {HIGH_VOL_MULT}",
            "C": "not A/B/D",
            "D": f"ATR14 < prior {ATR_AVG_DAYS} ATR14 average * {LOW_VOL_MULT} and body/range <= 0.35",
            "priority": "A overrides B/D; then B; then D; else C",
            "clean_filter": f"exclude abs(net) < {NOISE_ABS_NET}",
        },
        "active_plan_sha256_before": plan_before,
        "active_plan_sha256_after": active_plan_hash(),
    }
    diagnostics["active_plan_unchanged"] = diagnostics["active_plan_sha256_before"] == diagnostics["active_plan_sha256_after"]
    with (OUT_DIR / "diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    print(json.dumps({"out_dir": str(OUT_DIR), "summary_by_regime": regime_rows, "active_plan_unchanged": diagnostics["active_plan_unchanged"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
