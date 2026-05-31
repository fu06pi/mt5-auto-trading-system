#!/usr/bin/env python3
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

INPUT_CSV = Path(
    "/home/chain4655/Documents/backtest_reports/twitter_xauusd_candidates/"
    "20260520_225323/input_xauusd_m5.csv"
)
OUT_ROOT = Path("/home/chain4655/Documents/backtest_reports/london_asian_breakout_v2")
SYMBOL = "XAUUSD"
INITIAL_EQUITY = 10000.0
CONTRACT_SIZE = 100.0
MIN_LOT = 0.01
MAX_LOT = 1.2
RISK_PCT = 0.005
COMMISSION_PER_LOT_ROUND = 6.0
SPREAD_PRICE_COST = 0.40


@dataclasses.dataclass(frozen=True)
class Config:
    name: str
    mode: str  # first_break | confirmed | retest
    sides: str  # both | buy | sell
    start_hour: int
    start_minute: int
    end_hour: int
    end_minute: int
    rr: float
    max_hold_bars: int
    min_range: float
    max_range: float
    min_range_atr: float
    max_range_atr: float
    body_atr: float
    body_range: float
    close_quartile: float
    retest_bars: int
    rejection_body_atr: float
    use_ema_bias: bool = False
    skip_tuesday: bool = False
    friday_only: bool = False
    breakeven_at_r: float = 0.0


@dataclasses.dataclass(frozen=True)
class Entry:
    idx: int
    side: str
    sl: float
    tp: float
    setup_time: dt.datetime
    asia_hi: float
    asia_lo: float
    asia_range: float
    atr: float
    notes: str


@dataclasses.dataclass(frozen=True)
class Trade:
    config: str
    period: str
    entry_time: dt.datetime
    exit_time: dt.datetime
    side: str
    entry: float
    exit: float
    sl: float
    tp: float
    lots: float
    gross_pnl: float
    commission: float
    spread_cost: float
    net_pnl: float
    r_multiple: float
    reason: str
    bars_held: int
    asia_range: float
    atr: float
    entry_hour: int
    dow: int
    month: str
    notes: str


def load_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)
    df["time"] = pd.to_datetime(df["time"])
    needed = {"atr14", "ema20", "ema50", "date", "hour", "minute", "dow"}
    if not needed.issubset(df.columns):
        df = add_indicators(df)
    else:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values("time").reset_index(drop=True)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr14"] = tr.rolling(14).mean()
    out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["ema50"] = out["close"].ewm(span=50, adjust=False).mean()
    out["date"] = out["time"].dt.date
    out["hour"] = out["time"].dt.hour
    out["minute"] = out["time"].dt.minute
    out["dow"] = out["time"].dt.dayofweek
    return out


def minute_of_day(row: pd.Series) -> int:
    return int(row.hour) * 60 + int(row.minute)


def within_window(row: pd.Series, cfg: Config) -> bool:
    minute = minute_of_day(row)
    start = cfg.start_hour * 60 + cfg.start_minute
    end = cfg.end_hour * 60 + cfg.end_minute
    return start <= minute <= end


def candle_body(row: pd.Series) -> float:
    return abs(float(row.close) - float(row.open))


def close_position_fraction(row: pd.Series, side: str) -> float:
    high = float(row.high)
    low = float(row.low)
    if high <= low:
        return 0.5
    if side == "BUY":
        return (float(row.close) - low) / (high - low)
    return (high - float(row.close)) / (high - low)


def direction_allowed(side: str, cfg: Config) -> bool:
    return cfg.sides == "both" or (cfg.sides == "buy" and side == "BUY") or (
        cfg.sides == "sell" and side == "SELL"
    )


def position_size(equity: float, entry: float, sl: float) -> float:
    risk_amount = equity * RISK_PCT
    risk_per_lot = abs(entry - sl) * CONTRACT_SIZE
    if risk_per_lot <= 0:
        return 0.0
    lots = math.floor(min(MAX_LOT, risk_amount / risk_per_lot) / MIN_LOT) * MIN_LOT
    return max(MIN_LOT, round(lots, 2)) if lots >= MIN_LOT else 0.0


def make_trade(
    config: str,
    period: str,
    side: str,
    entry_time: dt.datetime,
    exit_time: dt.datetime,
    entry: float,
    exit_price: float,
    sl: float,
    tp: float,
    lots: float,
    reason: str,
    bars_held: int,
    asia_range: float,
    atr: float,
    notes: str,
) -> Trade:
    mult = 1.0 if side == "BUY" else -1.0
    gross = (exit_price - entry) * mult * lots * CONTRACT_SIZE
    commission = -COMMISSION_PER_LOT_ROUND * lots
    spread_cost = -SPREAD_PRICE_COST * lots * CONTRACT_SIZE
    net = gross + commission + spread_cost
    risk_amount = abs(entry - sl) * lots * CONTRACT_SIZE
    return Trade(
        config=config,
        period=period,
        entry_time=entry_time,
        exit_time=exit_time,
        side=side,
        entry=round(entry, 2),
        exit=round(exit_price, 2),
        sl=round(sl, 2),
        tp=round(tp, 2),
        lots=lots,
        gross_pnl=round(gross, 2),
        commission=round(commission, 2),
        spread_cost=round(spread_cost, 2),
        net_pnl=round(net, 2),
        r_multiple=round(net / max(risk_amount, 1e-9), 3),
        reason=reason,
        bars_held=bars_held,
        asia_range=round(asia_range, 2),
        atr=round(atr, 3),
        entry_hour=entry_time.hour,
        dow=entry_time.weekday(),
        month=entry_time.strftime("%Y-%m"),
        notes=notes,
    )


def candidate_stop_target(side: str, row: pd.Series, hi: float, lo: float, rng: float, atr: float, cfg: Config) -> Tuple[float, float]:
    close = float(row.close)
    stop_dist = max(atr * 1.2, rng * 0.55)
    if side == "BUY":
        sl = min(lo, close - stop_dist)
        tp = close + (close - sl) * cfg.rr
    else:
        sl = max(hi, close + stop_dist)
        tp = close - (sl - close) * cfg.rr
    return sl, tp


def trend_bias_ok(side: str, row: pd.Series, cfg: Config) -> bool:
    if not cfg.use_ema_bias:
        return True
    if side == "BUY":
        return float(row.ema20) >= float(row.ema50) and float(row.close) >= float(row.ema50)
    return float(row.ema20) <= float(row.ema50) and float(row.close) <= float(row.ema50)


def breakout_quality_ok(side: str, row: pd.Series, rng: float, atr: float, cfg: Config) -> bool:
    body = candle_body(row)
    if cfg.body_atr > 0 and body < atr * cfg.body_atr:
        return False
    if cfg.body_range > 0 and body < rng * cfg.body_range:
        return False
    if cfg.close_quartile > 0 and close_position_fraction(row, side) < cfg.close_quartile:
        return False
    return True


def generate_entries(df: pd.DataFrame, cfg: Config) -> List[Entry]:
    entries: List[Entry] = []
    for _day, g in df.groupby("date", sort=True):
        if g.empty:
            continue
        if cfg.skip_tuesday and int(g.iloc[0].dow) == 1:
            continue
        if cfg.friday_only and int(g.iloc[0].dow) != 4:
            continue
        asia = g[(g.hour >= 0) & (g.hour < 7)]
        london = g[g.apply(lambda row: within_window(row, cfg), axis=1)]
        if len(asia) < 24 or london.empty:
            continue
        hi = float(asia.high.max())
        lo = float(asia.low.min())
        rng = hi - lo
        asia_atr = float(asia.atr14.dropna().tail(12).mean()) if not asia.atr14.dropna().empty else 0.0
        range_atr = rng / max(asia_atr, 1e-9)
        if rng < cfg.min_range or rng > cfg.max_range:
            continue
        if range_atr < cfg.min_range_atr or range_atr > cfg.max_range_atr:
            continue
        if cfg.mode in {"first_break", "confirmed"}:
            entry = first_or_confirmed_entry(df, london, hi, lo, rng, cfg)
        elif cfg.mode == "retest":
            entry = retest_entry(df, london, hi, lo, rng, cfg)
        else:
            raise ValueError(f"Unknown mode: {cfg.mode}")
        if entry is not None:
            entries.append(entry)
    return entries


def first_or_confirmed_entry(df: pd.DataFrame, london: pd.DataFrame, hi: float, lo: float, rng: float, cfg: Config) -> Optional[Entry]:
    for idx, row in london.iterrows():
        if idx <= 0:
            continue
        prev = df.iloc[idx - 1]
        atr = float(row.atr14)
        if math.isnan(atr) or atr <= 0:
            continue
        side = ""
        if float(row.close) > hi and float(prev.close) <= hi:
            side = "BUY"
        elif float(row.close) < lo and float(prev.close) >= lo:
            side = "SELL"
        if not side or not direction_allowed(side, cfg) or not trend_bias_ok(side, row, cfg):
            continue
        if cfg.mode == "confirmed" and not breakout_quality_ok(side, row, rng, atr, cfg):
            continue
        sl, tp = candidate_stop_target(side, row, hi, lo, rng, atr, cfg)
        return Entry(
            idx=int(idx),
            side=side,
            sl=sl,
            tp=tp,
            setup_time=row.time.to_pydatetime(),
            asia_hi=hi,
            asia_lo=lo,
            asia_range=rng,
            atr=atr,
            notes=(
                f"mode={cfg.mode};asia_hi={hi:.2f};asia_lo={lo:.2f};range_atr={rng / max(atr, 1e-9):.2f};"
                f"body={candle_body(row):.2f};close_frac={close_position_fraction(row, side):.2f}"
            ),
        )
    return None


def retest_entry(df: pd.DataFrame, london: pd.DataFrame, hi: float, lo: float, rng: float, cfg: Config) -> Optional[Entry]:
    broke_side = ""
    broke_idx: Optional[int] = None
    broke_atr = 0.0
    for idx, row in london.iterrows():
        if idx <= 0:
            continue
        atr = float(row.atr14)
        if math.isnan(atr) or atr <= 0:
            continue
        if not broke_side:
            prev = df.iloc[idx - 1]
            if float(row.close) > hi and float(prev.close) <= hi:
                candidate_side = "BUY"
            elif float(row.close) < lo and float(prev.close) >= lo:
                candidate_side = "SELL"
            else:
                continue
            if not direction_allowed(candidate_side, cfg) or not trend_bias_ok(candidate_side, row, cfg):
                continue
            if not breakout_quality_ok(candidate_side, row, rng, atr, cfg):
                continue
            broke_side = candidate_side
            broke_idx = int(idx)
            broke_atr = atr
            continue
        if broke_idx is None or int(idx) - broke_idx > cfg.retest_bars:
            return None
        # Retest and rejection: wick/touch box edge, then close back in breakout direction.
        body = candle_body(row)
        if body < broke_atr * cfg.rejection_body_atr:
            continue
        if broke_side == "BUY":
            touched = float(row.low) <= hi
            rejected = float(row.close) > hi and float(row.close) > float(row.open)
        else:
            touched = float(row.high) >= lo
            rejected = float(row.close) < lo and float(row.close) < float(row.open)
        if not touched or not rejected:
            continue
        sl, tp = candidate_stop_target(broke_side, row, hi, lo, rng, atr, cfg)
        return Entry(
            idx=int(idx),
            side=broke_side,
            sl=sl,
            tp=tp,
            setup_time=row.time.to_pydatetime(),
            asia_hi=hi,
            asia_lo=lo,
            asia_range=rng,
            atr=atr,
            notes=(
                f"mode=retest;broke_idx={broke_idx};asia_hi={hi:.2f};asia_lo={lo:.2f};"
                f"range_atr={rng / max(atr, 1e-9):.2f};retest_body={body:.2f}"
            ),
        )
    return None


def simulate_entries(df: pd.DataFrame, period: str, cfg: Config, entries: Sequence[Entry]) -> List[Trade]:
    trades: List[Trade] = []
    equity = INITIAL_EQUITY
    last_exit_idx = -1
    traded_dates: set[object] = set()
    for entry_def in sorted(entries, key=lambda x: x.idx):
        idx = entry_def.idx
        if idx <= last_exit_idx or idx >= len(df) - 2:
            continue
        entry_idx = idx + 1
        entry_bar = df.iloc[entry_idx]
        if entry_bar.date in traded_dates:
            continue
        side = entry_def.side
        entry = float(entry_bar.open)
        sl = entry_def.sl
        tp = entry_def.tp
        if side == "BUY" and not (sl < entry < tp):
            continue
        if side == "SELL" and not (tp < entry < sl):
            continue
        lots = position_size(equity, entry, sl)
        if lots <= 0:
            continue
        exit_price = float(df.iloc[min(entry_idx + cfg.max_hold_bars, len(df) - 1)].close)
        exit_idx = min(entry_idx + cfg.max_hold_bars, len(df) - 1)
        reason = "TIME"
        active_sl = sl
        moved_be = False
        risk = abs(entry - sl)
        for j in range(entry_idx, min(entry_idx + cfg.max_hold_bars + 1, len(df))):
            bar = df.iloc[j]
            if cfg.breakeven_at_r > 0 and not moved_be:
                if side == "BUY" and float(bar.high) >= entry + risk * cfg.breakeven_at_r:
                    active_sl = entry
                    moved_be = True
                elif side == "SELL" and float(bar.low) <= entry - risk * cfg.breakeven_at_r:
                    active_sl = entry
                    moved_be = True
            if side == "BUY":
                hit_sl = float(bar.low) <= active_sl
                hit_tp = float(bar.high) >= tp
            else:
                hit_sl = float(bar.high) >= active_sl
                hit_tp = float(bar.low) <= tp
            if hit_sl and hit_tp:
                exit_price, exit_idx, reason = active_sl, j, "SL_FIRST_BE" if moved_be else "SL_FIRST"
                break
            if hit_sl:
                exit_price, exit_idx, reason = active_sl, j, "BE" if moved_be and active_sl == entry else "SL"
                break
            if hit_tp:
                exit_price, exit_idx, reason = tp, j, "TP"
                break
        trade = make_trade(
            config=cfg.name,
            period=period,
            side=side,
            entry_time=entry_bar.time.to_pydatetime(),
            exit_time=df.iloc[exit_idx].time.to_pydatetime(),
            entry=entry,
            exit_price=exit_price,
            sl=sl,
            tp=tp,
            lots=lots,
            reason=reason,
            bars_held=exit_idx - entry_idx,
            asia_range=entry_def.asia_range,
            atr=entry_def.atr,
            notes=entry_def.notes,
        )
        trades.append(trade)
        equity += trade.net_pnl
        last_exit_idx = exit_idx + 3
        traded_dates.add(entry_bar.date)
    return trades


def metrics(trades: Sequence[Trade]) -> Dict[str, object]:
    equity = INITIAL_EQUITY
    peak = equity
    max_dd = 0.0
    for trade in sorted(trades, key=lambda item: item.exit_time):
        equity += trade.net_pnl
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1.0)
    wins = [trade.net_pnl for trade in trades if trade.net_pnl > 0]
    losses = [trade.net_pnl for trade in trades if trade.net_pnl <= 0]
    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    return {
        "trades": len(trades),
        "net_pnl": round(equity - INITIAL_EQUITY, 2),
        "return_pct": round((equity / INITIAL_EQUITY - 1.0) * 100, 3),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 2) if trades else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 3)
        if gross_loss > 0
        else (999.0 if gross_profit > 0 else 0.0),
        "max_dd_pct": round(max_dd * 100, 3),
        "expectancy_usd": round(statistics.fmean([trade.net_pnl for trade in trades]), 2)
        if trades
        else 0.0,
        "avg_r": round(statistics.fmean([trade.r_multiple for trade in trades]), 3) if trades else 0.0,
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(-gross_loss, 2),
    }


def group_stats(trades: Sequence[Trade], key: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    values = sorted({getattr(trade, key) for trade in trades})
    for value in values:
        group = [trade for trade in trades if getattr(trade, key) == value]
        rows.append({"group_key": key, "group_value": value, **metrics(group)})
    return rows


def range_bucket(value: float) -> str:
    if value < 20:
        return "lt20"
    if value < 25:
        return "20_25"
    if value < 30:
        return "25_30"
    if value < 35:
        return "30_35"
    return "gte35"


def add_custom_splits(trades: Sequence[Trade]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for side in sorted({trade.side for trade in trades}):
        rows.append({"group_key": "side", "group_value": side, **metrics([t for t in trades if t.side == side])})
    for hour in sorted({trade.entry_hour for trade in trades}):
        rows.append({"group_key": "entry_hour", "group_value": hour, **metrics([t for t in trades if t.entry_hour == hour])})
    for month in sorted({trade.month for trade in trades}):
        rows.append({"group_key": "month", "group_value": month, **metrics([t for t in trades if t.month == month])})
    for bucket in ["lt20", "20_25", "25_30", "30_35", "gte35"]:
        group = [trade for trade in trades if range_bucket(trade.asia_range) == bucket]
        if group:
            rows.append({"group_key": "asia_range", "group_value": bucket, **metrics(group)})
    return rows


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fields: Optional[List[str]] = None) -> None:
    row_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(row_list[0].keys()) if row_list else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row_list)


def configs() -> List[Config]:
    base = Config(
        name="baseline_first_break_rr2_h48",
        mode="first_break",
        sides="both",
        start_hour=7,
        start_minute=0,
        end_hour=10,
        end_minute=59,
        rr=2.0,
        max_hold_bars=48,
        min_range=2.0,
        max_range=35.0,
        min_range_atr=0.0,
        max_range_atr=99.0,
        body_atr=0.0,
        body_range=0.0,
        close_quartile=0.0,
        retest_bars=0,
        rejection_body_atr=0.0,
    )
    out = [base]
    for rr in [1.5, 2.0, 2.5, 3.0]:
        for hold in [24, 36, 48, 60]:
            out.append(dataclasses.replace(base, name=f"buy_only_rr{rr}_h{hold}", sides="buy", rr=rr, max_hold_bars=hold))
            out.append(
                dataclasses.replace(
                    base,
                    name=f"buy_0800_0930_rr{rr}_h{hold}",
                    sides="buy",
                    start_hour=8,
                    start_minute=0,
                    end_hour=9,
                    end_minute=30,
                    rr=rr,
                    max_hold_bars=hold,
                )
            )
            out.append(
                dataclasses.replace(
                    base,
                    name=f"confirmed_buy_rr{rr}_h{hold}",
                    mode="confirmed",
                    sides="buy",
                    rr=rr,
                    max_hold_bars=hold,
                    body_atr=0.35,
                    body_range=0.12,
                    close_quartile=0.65,
                )
            )
            out.append(
                dataclasses.replace(
                    base,
                    name=f"confirmed_buy_0800_0930_rr{rr}_h{hold}",
                    mode="confirmed",
                    sides="buy",
                    start_hour=8,
                    start_minute=0,
                    end_hour=9,
                    end_minute=30,
                    rr=rr,
                    max_hold_bars=hold,
                    body_atr=0.35,
                    body_range=0.12,
                    close_quartile=0.65,
                )
            )
            out.append(
                dataclasses.replace(
                    base,
                    name=f"retest_buy_rr{rr}_h{hold}",
                    mode="retest",
                    sides="buy",
                    rr=rr,
                    max_hold_bars=hold,
                    body_atr=0.25,
                    body_range=0.08,
                    close_quartile=0.55,
                    retest_bars=12,
                    rejection_body_atr=0.15,
                )
            )
            out.append(
                dataclasses.replace(
                    base,
                    name=f"confirmed_both_0800_0930_rr{rr}_h{hold}",
                    mode="confirmed",
                    sides="both",
                    start_hour=8,
                    start_minute=0,
                    end_hour=9,
                    end_minute=30,
                    rr=rr,
                    max_hold_bars=hold,
                    body_atr=0.35,
                    body_range=0.12,
                    close_quartile=0.65,
                )
            )
    # Range/weekday hypotheses layered only on strong simple families, to avoid exploding grid.
    for rr in [1.5, 2.0, 2.5]:
        out.append(
            dataclasses.replace(
                base,
                name=f"buy_range20_35_rr{rr}_h48",
                sides="buy",
                rr=rr,
                min_range=20.0,
                max_range=35.0,
            )
        )
        out.append(
            dataclasses.replace(
                base,
                name=f"buy_skip_tue_rr{rr}_h48",
                sides="buy",
                rr=rr,
                skip_tuesday=True,
            )
        )
        out.append(
            dataclasses.replace(
                base,
                name=f"buy_ema_bias_rr{rr}_h48",
                sides="buy",
                rr=rr,
                use_ema_bias=True,
            )
        )
        out.append(
            dataclasses.replace(
                base,
                name=f"buy_be1r_rr{rr}_h48",
                sides="buy",
                rr=rr,
                breakeven_at_r=1.0,
            )
        )
    return out


def monthly_walkforward(raw: pd.DataFrame, cfg: Config) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    df = raw[raw.time >= raw.time.max() - pd.Timedelta(days=184)].reset_index(drop=True)
    all_entries = generate_entries(df, cfg)
    all_trades = simulate_entries(df, "6m", cfg, all_entries)
    for month in sorted({trade.month for trade in all_trades}):
        month_trades = [trade for trade in all_trades if trade.month == month]
        rows.append({"config": cfg.name, "month": month, **metrics(month_trades)})
    return rows


def main() -> None:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = load_data()
    periods = {"3m": 92, "6m": 184}
    summary_rows: List[Dict[str, object]] = []
    trade_rows: List[Dict[str, object]] = []
    split_rows: List[Dict[str, object]] = []
    cfg_rows = [dataclasses.asdict(cfg) for cfg in configs()]
    write_csv(out_dir / "config_grid.csv", cfg_rows)
    for period, days in periods.items():
        df = raw[raw.time >= raw.time.max() - pd.Timedelta(days=days)].reset_index(drop=True)
        for cfg in configs():
            entries = generate_entries(df, cfg)
            trades = simulate_entries(df, period, cfg, entries)
            m = metrics(trades)
            summary_rows.append({"period": period, "config": cfg.name, **m, **dataclasses.asdict(cfg)})
            for trade in trades:
                row = dataclasses.asdict(trade)
                row["entry_time"] = trade.entry_time.isoformat()
                row["exit_time"] = trade.exit_time.isoformat()
                trade_rows.append(row)
            for split in add_custom_splits(trades):
                split_rows.append({"period": period, "config": cfg.name, **split})
    ranked = sorted(
        summary_rows,
        key=lambda row: (
            int(row["period"] == "6m"),
            float(row["profit_factor"]),
            float(row["return_pct"]),
            int(row["trades"]),
        ),
        reverse=True,
    )
    viable = [
        row
        for row in summary_rows
        if row["period"] == "6m"
        and int(row["trades"]) >= 8
        and float(row["profit_factor"]) >= 1.2
        and float(row["expectancy_usd"]) > 0
    ]
    viable_ranked = sorted(
        viable,
        key=lambda row: (float(row["profit_factor"]), float(row["return_pct"]), -abs(float(row["max_dd_pct"]))),
        reverse=True,
    )
    write_csv(out_dir / "metrics_comparison.csv", summary_rows)
    write_csv(out_dir / "ranked_all.csv", ranked)
    write_csv(out_dir / "ranked_viable_6m.csv", viable_ranked)
    write_csv(out_dir / "all_trades.csv", trade_rows)
    write_csv(out_dir / "split_stats.csv", split_rows)
    monthly_rows: List[Dict[str, object]] = []
    for row in viable_ranked[:10]:
        cfg = next(item for item in configs() if item.name == row["config"])
        monthly_rows.extend(monthly_walkforward(raw, cfg))
    write_csv(out_dir / "top10_monthly_walkforward.csv", monthly_rows)
    report = build_report(out_dir, summary_rows, viable_ranked, split_rows, monthly_rows)
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "generated_at": dt.datetime.now().isoformat(sep=" "),
                "symbol": SYMBOL,
                "input_csv": str(INPUT_CSV),
                "out_dir": str(out_dir),
                "top_viable_6m": viable_ranked[:20],
                "assumptions": {
                    "initial_equity": INITIAL_EQUITY,
                    "risk_pct": RISK_PCT,
                    "max_lots": MAX_LOT,
                    "commission_per_lot_round": COMMISSION_PER_LOT_ROUND,
                    "spread_price_cost": SPREAD_PRICE_COST,
                    "fill_model": "next-bar open; SL-first if SL/TP same M5 bar; max 1 trade/day",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(out_dir), "top_viable_6m": viable_ranked[:10]}, ensure_ascii=False, indent=2))


def build_report(
    out_dir: Path,
    summary_rows: Sequence[Dict[str, object]],
    viable_ranked: Sequence[Dict[str, object]],
    split_rows: Sequence[Dict[str, object]],
    monthly_rows: Sequence[Dict[str, object]],
) -> str:
    baseline = [row for row in summary_rows if row["config"] == "baseline_first_break_rr2_h48"]
    lines = [
        "# London Asian Range Breakout v2 Optimization",
        "",
        f"Output: `{out_dir}`",
        "",
        "## Assumptions",
        f"- Initial equity: {INITIAL_EQUITY}",
        f"- Risk per trade: {RISK_PCT * 100:.2f}%",
        f"- Commission: ${COMMISSION_PER_LOT_ROUND}/lot round-turn",
        f"- Spread cost: {SPREAD_PRICE_COST:.2f} XAUUSD price units",
        "- Entry at next M5 open; conservative SL-first if SL and TP both touched in same bar.",
        "- Max 1 trade/day.",
        "",
        "## Baseline check",
    ]
    for row in baseline:
        lines.append(
            f"- {row['period']}: trades={row['trades']}, return={row['return_pct']}%, "
            f"PF={row['profit_factor']}, DD={row['max_dd_pct']}%, exp=${row['expectancy_usd']}"
        )
    lines += ["", "## Top viable 6m configs"]
    for row in viable_ranked[:15]:
        lines.append(
            f"- {row['config']}: trades={row['trades']}, return={row['return_pct']}%, "
            f"PF={row['profit_factor']}, DD={row['max_dd_pct']}%, exp=${row['expectancy_usd']}, "
            f"win={row['win_rate_pct']}%, rr={row['rr']}, hold={row['max_hold_bars']}, "
            f"mode={row['mode']}, sides={row['sides']}, window={row['start_hour']:02d}:{row['start_minute']:02d}-{row['end_hour']:02d}:{row['end_minute']:02d}"
        )
    if viable_ranked:
        top_name = str(viable_ranked[0]["config"])
        lines += ["", f"## Split stats for top config: {top_name}"]
        top_splits = [row for row in split_rows if row["period"] == "6m" and row["config"] == top_name]
        for row in top_splits:
            lines.append(
                f"- {row['group_key']}={row['group_value']}: trades={row['trades']}, "
                f"return={row['return_pct']}%, PF={row['profit_factor']}, exp=${row['expectancy_usd']}"
            )
        lines += ["", "## Monthly walk-forward for top configs"]
        for row in monthly_rows[:80]:
            lines.append(
                f"- {row['config']} {row['month']}: trades={row['trades']}, return={row['return_pct']}%, "
                f"PF={row['profit_factor']}, exp=${row['expectancy_usd']}"
            )
    lines += [
        "",
        "## Files",
        "- `metrics_comparison.csv`: full grid results.",
        "- `ranked_viable_6m.csv`: 6m configs with trades>=8, PF>=1.2, positive expectancy.",
        "- `all_trades.csv`: every simulated trade.",
        "- `split_stats.csv`: side/hour/month/range-bucket splits.",
        "- `top10_monthly_walkforward.csv`: monthly stability for top configs.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
