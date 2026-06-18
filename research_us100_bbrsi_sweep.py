from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from pymt5linux import MetaTrader5

OUT = Path('/home/chain4655/Documents/Projects/MT5/backtest_reports_us100_bbrsi')
OUT.mkdir(parents=True, exist_ok=True)
SYMBOL = 'US100.cash'
SPREAD = 2.0  # current observed spread ~1.95 index points


@dataclass(frozen=True)
class Trade:
    name: str
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    r: float
    reason: str


def fetch_mt5_bars() -> pd.DataFrame:
    mt5 = MetaTrader5(host='127.0.0.1', port=18812)
    if not mt5.initialize():
        raise RuntimeError(f'MT5 initialize failed: {mt5.last_error()}')
    mt5.symbol_select(SYMBOL, True)
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, 50000)
    mt5.shutdown()
    if rates is None or len(rates) == 0:
        raise RuntimeError('No US100 bars from MT5')
    rows = []
    for row in list(rates):
        def get(name: str, idx: int) -> float:
            try:
                return float(row[name])
            except Exception:
                pass
            if hasattr(row, name):
                return float(getattr(row, name))
            return float(row[idx])
        rows.append({
            'time': int(get('time', 0)),
            'open': get('open', 1),
            'high': get('high', 2),
            'low': get('low', 3),
            'close': get('close', 4),
            'tick_volume': get('tick_volume', 5) if len(row) > 5 else 0.0,
        })
    df = pd.DataFrame(rows)
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    return df[['datetime', 'open', 'high', 'low', 'close', 'tick_volume']].astype({
        'open': float, 'high': float, 'low': float, 'close': float, 'tick_volume': float
    })


def indicators(df: pd.DataFrame, bb_period: int, bb_std: float, rsi_period: int, ema_period: int, slope_lookback: int) -> pd.DataFrame:
    out = df.copy()
    prev_close = out.close.shift(1)
    tr = pd.concat([(out.high - out.low), (out.high - prev_close).abs(), (out.low - prev_close).abs()], axis=1).max(axis=1)
    out['atr'] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    up_move = out.high.diff()
    down_move = -out.low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr_sm = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / tr_sm
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / tr_sm
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    out['adx'] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    delta = out.close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    out['rsi'] = 100 - 100 / (1 + gain / loss)
    out['bb_mid'] = out.close.rolling(bb_period).mean()
    out['bb_std'] = out.close.rolling(bb_period).std()
    out['bb_upper'] = out.bb_mid + bb_std * out.bb_std
    out['bb_lower'] = out.bb_mid - bb_std * out.bb_std
    out['ema'] = out.close.ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
    out['ema_slope_atr'] = (out.ema - out.ema.shift(slope_lookback)).abs() / out.atr.clip(lower=1e-9)
    out['atr_med_20d'] = out.atr.rolling(5760, min_periods=1000).median()
    out['atr_ratio_med'] = out.atr / out.atr_med_20d
    return out


def entry_signal(row: object, cfg: dict) -> Optional[dict]:
    if pd.isna(row.adx) or pd.isna(row.bb_lower) or pd.isna(row.atr_ratio_med):
        return None
    if row.adx > cfg['adx_max'] or row.ema_slope_atr > cfg['slope_max']:
        return None
    if not (cfg['atr_min'] <= row.atr_ratio_med <= cfg['atr_max']):
        return None
    if row.low <= row.bb_lower and row.close > row.bb_lower and row.rsi <= cfg['rsi_lo']:
        return {'side': 'long', 'stop': row.low - cfg['stop_atr'] * row.atr, 'target': row.bb_upper}
    if row.high >= row.bb_upper and row.close < row.bb_upper and row.rsi >= cfg['rsi_hi']:
        return {'side': 'short', 'stop': row.high + cfg['stop_atr'] * row.atr, 'target': row.bb_lower}
    return None


def opposite_signal(row: object, side: str, cfg: dict) -> bool:
    if side == 'long':
        return row.high >= row.bb_upper and row.close < row.bb_upper and row.rsi >= cfg['rsi_hi']
    return row.low <= row.bb_lower and row.close > row.bb_lower and row.rsi <= cfg['rsi_lo']


def simulate(df: pd.DataFrame, cfg: dict) -> list[Trade]:
    trades: list[Trade] = []
    pos: Optional[dict] = None
    pending: Optional[dict] = None
    hs = SPREAD / 2
    for i, row in enumerate(df.itertuples(index=False)):
        if i < 6000:
            continue
        if pending and pos is None:
            side = pending['side']
            entry = row.open + hs if side == 'long' else row.open - hs
            stop = pending['stop']
            target = pending['target']
            risk = entry - stop if side == 'long' else stop - entry
            if risk > 0:
                pos = {**pending, 'entry': entry, 'entry_i': i, 'entry_time': row.datetime, 'risk': risk, 'stop': stop, 'target': target}
            pending = None
        if pos is not None:
            side = pos['side']
            reason = ''
            exit_price = None
            if side == 'long':
                if row.low <= pos['stop']:
                    exit_price = pos['stop'] - hs; reason = 'sl'
                elif row.high >= pos['target']:
                    exit_price = pos['target'] - hs; reason = 'tp'
                elif cfg['signal_flip'] and opposite_signal(row, side, cfg):
                    exit_price = row.close - hs; reason = 'signal_flip'
                elif i - pos['entry_i'] >= cfg['max_hold']:
                    exit_price = row.close - hs; reason = 'max_hold'
                if exit_price is not None:
                    r = (exit_price - pos['entry']) / pos['risk']
            else:
                if row.high >= pos['stop']:
                    exit_price = pos['stop'] + hs; reason = 'sl'
                elif row.low <= pos['target']:
                    exit_price = pos['target'] + hs; reason = 'tp'
                elif cfg['signal_flip'] and opposite_signal(row, side, cfg):
                    exit_price = row.close + hs; reason = 'signal_flip'
                elif i - pos['entry_i'] >= cfg['max_hold']:
                    exit_price = row.close + hs; reason = 'max_hold'
                if exit_price is not None:
                    r = (pos['entry'] - exit_price) / pos['risk']
            if exit_price is not None:
                trades.append(Trade(cfg['name'], side, pos['entry_time'], row.datetime, float(r), reason))
                pos = None
            continue
        sig = entry_signal(row, cfg)
        if sig is not None:
            pending = sig
    return trades


def summarize(trades: list[Trade]) -> dict:
    if not trades:
        return {'trades': 0, 'net_r': 0.0, 'pf': 0.0, 'win_rate': 0.0, 'max_dd_r': 0.0}
    r = pd.Series([t.r for t in trades], dtype='float64')
    eq = r.cumsum()
    dd = eq - eq.cummax()
    gw = r[r > 0].sum()
    gl = -r[r < 0].sum()
    return {
        'trades': len(trades), 'net_r': round(float(r.sum()), 2), 'avg_r': round(float(r.mean()), 3),
        'pf': round(float(gw / gl), 3) if gl > 0 else 99.0,
        'win_rate': round(float((r > 0).mean() * 100), 1), 'max_dd_r': round(float(dd.min()), 2),
        'tp': int(sum(t.reason == 'tp' for t in trades)), 'sl': int(sum(t.reason == 'sl' for t in trades)),
        'time': int(sum(t.reason == 'max_hold' for t in trades)), 'signal': int(sum(t.reason == 'signal_flip' for t in trades)),
        'long': int(sum(t.side == 'long' for t in trades)), 'short': int(sum(t.side == 'short' for t in trades)),
    }


def main() -> None:
    raw = fetch_mt5_bars()
    raw.to_csv(OUT / 'us100_m5_bars.csv', index=False)
    rows = []
    for bb_period in [20, 30, 40]:
        for bb_std in [2.0, 2.2, 2.5]:
            for rsi_lo, rsi_hi in [(35, 65), (38, 62), (40, 60), (45, 55)]:
                df = indicators(raw, bb_period, bb_std, 14, 50, 1)
                for adx_max in [16, 20, 24, 28]:
                    for slope_max in [0.20, 0.35, 0.50, 0.80]:
                        for atr_min, atr_max in [(0.65, 1.65), (0.8, 1.5), (0.7, 1.35), (0.9, 1.8)]:
                            cfg = {
                                'name': f'bb{bb_period}_{bb_std}_rsi{rsi_lo}-{rsi_hi}_adx{adx_max}_s{slope_max}_atr{atr_min}-{atr_max}',
                                'bb_period': bb_period, 'bb_std': bb_std, 'rsi_lo': rsi_lo, 'rsi_hi': rsi_hi,
                                'adx_max': adx_max, 'slope_max': slope_max, 'atr_min': atr_min, 'atr_max': atr_max,
                                'max_hold': 36, 'signal_flip': True, 'stop_atr': 1.0,
                            }
                            trades = simulate(df, cfg)
                            s = summarize(trades)
                            s.update(cfg)
                            rows.append(s)
    res = pd.DataFrame(rows).sort_values(['pf', 'net_r'], ascending=[False, False])
    res.to_csv(OUT / 'us100_bbrsi_sweep.csv', index=False)
    viable = res[(res.trades >= 20) & (res.net_r > 0) & (res.pf > 1.05)].copy()
    viable.to_csv(OUT / 'us100_bbrsi_viable.csv', index=False)
    print('range', raw.datetime.min(), raw.datetime.max(), 'bars', len(raw), 'spread_assumption', SPREAD)
    print('viable', len(viable), 'of', len(res))
    print(viable.head(15).to_string(index=False))


if __name__ == '__main__':
    main()
