from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pymt5linux import MetaTrader5

ROOT = Path('/home/chain4655/Documents/Projects/MT5')
OUT_DIR = ROOT / 'reports' / 'today_trade_context'
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAGIC_NAME = {
    204573: 'XAU trend M1',
    26052032: 'US100 trend M1',
    26052033: 'BTC trend M1',
    26052034: 'XAU OCC M15',
    26052035: 'XAU BBRSI range M5',
    26052036: 'US100 BBRSI range M5',
    0: 'manual/other',
}
MAGIC_LOG = {
    204573: ROOT / 'auto_quant/logs/xauusd_trend_chop_conservative.log',
    26052032: ROOT / 'auto_quant/logs/us100_mirror_xauusd.log',
    26052033: ROOT / 'auto_quant/logs/btcusd_mirror_xauusd.log',
    26052034: ROOT / 'auto_quant/logs/xauusd_occ_open_close_cross.log',
    26052035: ROOT / 'auto_quant/logs/xauusd_bbrsi_range.log',
    26052036: ROOT / 'auto_quant/logs/us100_bbrsi_range_monitor.log',
}
TIMEFRAME = {
    204573: 'M1',
    26052032: 'M1',
    26052033: 'M1',
    26052034: 'M15',
    26052035: 'M5',
    26052036: 'M5',
}

# MT5 constants from wrapper are accessible on object, but values are stable.
TF_MAP = {
    'M1': 1,
    'M5': 5,
    'M15': 15,
}
# Strategy process logs are written by the Wine/MT5 process clock, which is 3h behind
# the deal/rate timestamps returned by pymt5linux on this host.
LOG_TIME_SHIFT = timedelta(hours=-3)


def dt(ts: int) -> datetime:
    return datetime.fromtimestamp(int(ts))


def deal_dict(d: Any) -> Dict[str, Any]:
    return {name: getattr(d, name) for name in d._asdict().keys()} if hasattr(d, '_asdict') else d.__dict__


def pair_deals(deals: List[Any]) -> List[Dict[str, Any]]:
    by_pos: Dict[int, Dict[str, Any]] = defaultdict(lambda: {'entries': [], 'exits': []})
    for d in deals:
        if not getattr(d, 'symbol', ''):
            continue
        entry = getattr(d, 'entry', None)
        if entry == 0:  # in
            by_pos[getattr(d, 'position_id')]['entries'].append(d)
        elif entry == 1:  # out
            by_pos[getattr(d, 'position_id')]['exits'].append(d)
    rows = []
    for pos_id, parts in by_pos.items():
        if not parts['entries'] or not parts['exits']:
            continue
        e = sorted(parts['entries'], key=lambda x: x.time)[0]
        xs = sorted(parts['exits'], key=lambda x: x.time)
        x = xs[-1]
        profit = sum(float(getattr(z, 'profit', 0) or 0) for z in xs)
        commission = sum(float(getattr(z, 'commission', 0) or 0) for z in parts['entries'] + xs)
        rows.append({
            'position_id': pos_id,
            'symbol': e.symbol,
            'magic': int(e.magic),
            'strategy': MAGIC_NAME.get(int(e.magic), str(e.magic)),
            'direction': 'BUY' if int(e.type) == 0 else 'SELL',
            'volume': float(e.volume),
            'open_time': dt(e.time),
            'close_time': dt(x.time),
            'minutes_held': round((dt(x.time) - dt(e.time)).total_seconds() / 60, 1),
            'open_price': float(e.price),
            'close_price': float(x.price),
            'profit': round(profit, 2),
            'commission': round(commission, 2),
            'net_profit': round(profit + commission, 2),
            'exit_reason': int(x.reason),
            'exit_comment': str(x.comment),
            'entry_order': int(e.order),
            'exit_order': int(x.order),
        })
    return sorted(rows, key=lambda r: r['open_time'])


def parse_log_time(line: str) -> Optional[datetime]:
    m = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
    if not m:
        return None
    return datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')


def load_log(path: Path) -> List[Tuple[datetime, str]]:
    if not path.exists():
        return []
    out = []
    with path.open(errors='ignore') as f:
        for line in f:
            t = parse_log_time(line)
            if t:
                out.append((t, line.strip()))
    return out

LOG_CACHE = {magic: load_log(path) for magic, path in MAGIC_LOG.items()}


def closest_lines(magic: int, t: datetime, minutes: int = 3, limit: int = 10) -> List[str]:
    lines = LOG_CACHE.get(magic, [])
    lo, hi = t - timedelta(minutes=minutes), t + timedelta(minutes=minutes)
    candidates = [s for ts, s in lines if lo <= ts <= hi]
    # Prefer decisive lines.
    key_words = ('Bar ', 'order_send', 'Quality filter', 'HTF lag', 'Loss-close', 'Signal reversal', 'SLTP modify')
    picked = [s for s in candidates if any(k in s for k in key_words)]
    return picked[-limit:]


def find_entry_bar(magic: int, t: datetime) -> str:
    lines = LOG_CACHE.get(magic, [])
    bars = [(ts, s) for ts, s in lines if ts <= t and ' Bar ' in s]
    if not bars:
        return ''
    return bars[-1][1]


def get_market(mt5: MetaTrader5, row: Dict[str, Any]) -> Dict[str, Any]:
    symbol = row['symbol']
    mt5.symbol_select(symbol, True)
    start = row['open_time'] - timedelta(hours=2)
    end = row['close_time'] + timedelta(minutes=45)
    tf = TIMEFRAME.get(row['magic'], 'M1')
    rates = None
    last_error = None
    for _attempt in range(4):
        try:
            mt5.symbol_select(symbol, True)
            rates = mt5.copy_rates_range(symbol, getattr(mt5, f'TIMEFRAME_{tf}'), start, end)
            last_error = mt5.last_error()
            if rates is not None and len(rates) > 0:
                break
            mt5.shutdown()
            mt5.initialize()
        except Exception as exc:
            last_error = exc
            try:
                mt5.shutdown()
                mt5.initialize()
            except Exception:
                pass
    if rates is None or len(rates) == 0:
        return {'market_error': str(last_error)}
    df = pd.DataFrame(rates)
    df['dt'] = [datetime.fromtimestamp(int(x)) for x in df['time']]  # MT5 deal/log local time basis
    in_trade = df[(df['dt'] >= row['open_time']) & (df['dt'] <= row['close_time'])]
    pre = df[df['dt'] < row['open_time']].tail(60)
    if in_trade.empty and tf != 'M1':
        # Very short trades on M5/M15 can open and close inside a bar whose timestamp is before entry.
        # Fall back to M1 broker records for execution-window MFE/MAE.
        try:
            rates_m1 = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start, end)
            if rates_m1 is not None and len(rates_m1) > 0:
                df = pd.DataFrame(rates_m1)
                df['dt'] = [datetime.fromtimestamp(int(x)) for x in df['time']]
                in_trade = df[(df['dt'] >= row['open_time']) & (df['dt'] <= row['close_time'])]
                pre = df[df['dt'] < row['open_time']].tail(60)
                tf = f'{tf}/M1-metrics'
        except Exception:
            pass
    if in_trade.empty:
        # Include the nearest enclosing/preceding bar as a last-resort record.
        nearest = df[df['dt'] <= row['close_time']].tail(1)
        in_trade = nearest
        if in_trade.empty:
            return {'market_error': 'no bars in trade window'}
    direction = row['direction']
    op = row['open_price']
    if direction == 'BUY':
        mfe = max(0.0, in_trade['high'].max() - op)
        mae = max(0.0, op - in_trade['low'].min())
        close_move = row['close_price'] - op
        pre_move_15 = (pre['close'].iloc[-1] - pre['close'].iloc[-16]) if len(pre) >= 16 else math.nan
        post_30 = df[df['dt'] > row['close_time']].head(30)
        post_move = (post_30['close'].iloc[-1] - row['close_price']) if len(post_30) else math.nan
    else:
        mfe = max(0.0, op - in_trade['low'].min())
        mae = max(0.0, in_trade['high'].max() - op)
        close_move = op - row['close_price']
        pre_move_15 = (pre['close'].iloc[-16] - pre['close'].iloc[-1]) if len(pre) >= 16 else math.nan
        post_30 = df[df['dt'] > row['close_time']].head(30)
        post_move = (row['close_price'] - post_30['close'].iloc[-1]) if len(post_30) else math.nan
    return {
        'tf': tf,
        'bars_in_trade': int(len(in_trade)),
        'entry_bar_time': str(in_trade['dt'].iloc[0]),
        'entry_bar_open': round(float(in_trade['open'].iloc[0]), 5),
        'entry_bar_high': round(float(in_trade['high'].iloc[0]), 5),
        'entry_bar_low': round(float(in_trade['low'].iloc[0]), 5),
        'entry_bar_close': round(float(in_trade['close'].iloc[0]), 5),
        'mfe_price': round(float(mfe), 5),
        'mae_price': round(float(mae), 5),
        'close_move_price': round(float(close_move), 5),
        'mfe_to_mae': round(float(mfe / mae), 3) if mae and not math.isnan(mae) else '',
        'pre_15bar_dir_move': round(float(pre_move_15), 5) if not math.isnan(pre_move_15) else '',
        'post_30bar_followthrough': round(float(post_move), 5) if not math.isnan(post_move) else '',
    }


def extract_fields(bar_line: str) -> Dict[str, Any]:
    fields = {}
    for key in ['htf','htf_comp','spread','momentum','m15_mom','score','signal','source','positions','foreign','adx','atr_ratio','rsi','block']:
        m = re.search(rf'{key}=([^ |]+)', bar_line)
        if m:
            fields[f'log_{key}'] = m.group(1)
    return fields


def main() -> None:
    mt5 = MetaTrader5(host='127.0.0.1', port=18812)
    print('initialize', mt5.initialize(), mt5.last_error())
    now = datetime.now()
    deals = list(mt5.history_deals_get(now - timedelta(days=1), now) or [])
    trades = pair_deals(deals)
    # exclude tiny commission/noise only if absolute net <4? Keep but mark; user asked every trade.
    enriched = []
    for r in trades:
        m = get_market(mt5, r)
        log_time = r['open_time'] + LOG_TIME_SHIFT
        exit_log_time = r['close_time'] + LOG_TIME_SHIFT
        bar = find_entry_bar(r['magic'], log_time)
        log_fields = extract_fields(bar)
        r2 = {**r, **m, **log_fields}
        r2['entry_bar_log'] = bar
        r2['entry_context_lines'] = '\n'.join(closest_lines(r['magic'], log_time, 3, 8))
        r2['exit_context_lines'] = '\n'.join(closest_lines(r['magic'], exit_log_time, 3, 8))
        enriched.append(r2)
    mt5.shutdown()
    csv_path = OUT_DIR / 'today_closed_trade_context.csv'
    md_path = OUT_DIR / 'today_closed_trade_context.md'
    keys = sorted({k for r in enriched for k in r.keys()})
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(enriched)
    with md_path.open('w', encoding='utf-8') as f:
        f.write('# Today closed trade context\n\n')
        f.write(f'Generated: {datetime.now()}\n\n')
        for r in enriched:
            f.write(f"## {r['open_time']} {r['strategy']} {r['symbol']} {r['direction']} pos={r['position_id']}\n")
            f.write(f"- close: {r['close_time']} reason={r['exit_reason']} comment={r['exit_comment']} net={r['net_profit']} held={r['minutes_held']}m\n")
            f.write(f"- price: open={r['open_price']} close={r['close_price']} MFE={r.get('mfe_price')} MAE={r.get('mae_price')} post30={r.get('post_30bar_followthrough')}\n")
            f.write(f"- log fields: signal={r.get('log_signal')} score={r.get('log_score')} htf={r.get('log_htf')} comp={r.get('log_htf_comp')} m15={r.get('log_m15_mom')} block={r.get('log_block')}\n")
            f.write('- entry bar log:\n```\n' + r.get('entry_bar_log','')[:2000] + '\n```\n')
            f.write('- entry context:\n```\n' + r.get('entry_context_lines','')[:3000] + '\n```\n')
            f.write('- exit context:\n```\n' + r.get('exit_context_lines','')[:3000] + '\n```\n\n')
    print('trades', len(enriched))
    print('csv', csv_path)
    print('md', md_path)
    # compact summary for terminal
    for r in enriched:
        if abs(float(r['net_profit'])) < 4:
            noise=' noise'
        else:
            noise=''
        print(f"{r['open_time'].strftime('%H:%M')} {r['strategy']} {r['symbol']} {r['direction']} net={r['net_profit']} mfe={r.get('mfe_price')} mae={r.get('mae_price')} signal={r.get('log_signal')} score={r.get('log_score')} htf={r.get('log_htf')}/{r.get('log_htf_comp')} m15={r.get('log_m15_mom')}{noise}")

if __name__ == '__main__':
    main()
