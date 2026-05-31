#!/usr/bin/env python3
"""Export MT5 live trade audit: deals, orders, closed-position PnL, equity curve, and log events."""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT = Path('/home/chain4655/Documents/Projects/MT5')
OUT_ROOT = Path('/home/chain4655/Documents/backtest_reports')
PYMT5_SITE = Path('/home/chain4655/.openharness-venv/lib64/python3.14/site-packages')
if str(PYMT5_SITE) not in sys.path:
    sys.path.insert(0, str(PYMT5_SITE))

try:
    from pymt5linux import MetaTrader5  # type: ignore
except ImportError as exc:
    raise SystemExit(f'pymt5linux import failed: {exc}')

TRADE_TYPES = {0: 'BUY', 1: 'SELL'}
ENTRY_MAP = {0: 'IN', 1: 'OUT', 2: 'INOUT', 3: 'OUT_BY'}


def rowdict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, '_asdict'):
        d = dict(obj._asdict())
    elif isinstance(obj, dict):
        d = dict(obj)
    else:
        d = {k: getattr(obj, k) for k in dir(obj) if not k.startswith('_')}
    for key in ('time', 'time_msc', 'time_setup', 'time_setup_msc', 'time_done', 'time_done_msc', 'expiration'):
        if key in d and d[key] not in (None, 0):
            try:
                sec = int(d[key]) / (1000 if key.endswith('_msc') else 1)
                d[key + '_iso'] = datetime.fromtimestamp(sec).isoformat(sep=' ')
            except (ValueError, OSError, TypeError):
                pass
    if 'type' in d:
        d['type_name'] = TRADE_TYPES.get(d.get('type'), str(d.get('type')))
    if 'entry' in d:
        d['entry_name'] = ENTRY_MAP.get(d.get('entry'), str(d.get('entry')))
    return d


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fields.append(k)
                seen.add(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def net_of_deal(d: Dict[str, Any]) -> float:
    return sum(float(d.get(k) or 0.0) for k in ('profit', 'commission', 'swap', 'fee'))


def max_drawdown(values: List[float]) -> tuple[float, float]:
    peak = -math.inf
    max_dd_abs = 0.0
    max_dd_pct = 0.0
    for v in values:
        if v > peak:
            peak = v
        if peak > 0:
            dd_abs = peak - v
            dd_pct = dd_abs / peak * 100
            if dd_abs > max_dd_abs:
                max_dd_abs = dd_abs
                max_dd_pct = dd_pct
    return max_dd_abs, max_dd_pct


def summarize_closed(closed: List[Dict[str, Any]], start_equity: float, end_equity: float) -> Dict[str, Any]:
    pnls = [float(r['net']) for r in closed]
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]
    gross_profit = sum(wins)
    gross_loss = sum(losses)
    eq = [start_equity]
    for x in pnls:
        eq.append(eq[-1] + x)
    dd_abs, dd_pct = max_drawdown(eq)
    return {
        'closed_positions': len(closed),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate_pct': round(len(wins) / len(pnls) * 100, 2) if pnls else 0.0,
        'gross_profit': round(gross_profit, 2),
        'gross_loss': round(gross_loss, 2),
        'net_pnl': round(sum(pnls), 2),
        'profit_factor': round(gross_profit / abs(gross_loss), 4) if gross_loss else None,
        'avg_win': round(sum(wins) / len(wins), 2) if wins else 0.0,
        'avg_loss': round(sum(losses) / len(losses), 2) if losses else 0.0,
        'expectancy': round(sum(pnls) / len(pnls), 2) if pnls else 0.0,
        'start_equity_est': round(start_equity, 2),
        'end_equity': round(end_equity, 2),
        'return_pct_est': round((end_equity - start_equity) / start_equity * 100, 4) if start_equity else None,
        'max_drawdown_abs_on_closed_curve': round(dd_abs, 2),
        'max_drawdown_pct_on_closed_curve': round(dd_pct, 4),
    }


def extract_log_events() -> List[Dict[str, Any]]:
    log_dir = PROJECT / 'auto_quant' / 'logs'
    patterns = [
        re.compile(r'(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:,\d+)? .*?(?P<event>ENTRY|EXIT|Closed deal|order_send|retcode|CLOSE_RESULT|ORDER_SEND).*', re.I),
        re.compile(r'(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:,\d+)? .*?Bar (?P<bar>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?signal=(?P<signal>BUY|SELL).*'),
    ]
    events: List[Dict[str, Any]] = []
    for path in sorted(log_dir.glob('*.log')):
        try:
            text = path.read_text(errors='ignore')
        except OSError:
            continue
        strategy = path.stem
        for line in text.splitlines():
            if not any(tok in line for tok in ['ENTRY', 'EXIT', 'Closed deal', 'order_send', 'retcode', 'CLOSE_RESULT', 'ORDER_SEND', 'signal=BUY', 'signal=SELL']):
                continue
            rec: Dict[str, Any] = {'source_file': str(path), 'strategy': strategy, 'line': line[:2000]}
            for pat in patterns:
                m = pat.search(line)
                if m:
                    rec.update({k: v for k, v in m.groupdict().items() if v is not None})
                    break
            events.append(rec)
    return events


def main() -> None:
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = OUT_ROOT / f'live_trade_audit_{stamp}'
    out.mkdir(parents=True, exist_ok=True)

    mt5 = MetaTrader5(host='127.0.0.1', port=18812)
    if not mt5.initialize():
        raise SystemExit(f'MT5 initialize failed: {mt5.last_error()}')
    ai = mt5.account_info()
    ti = mt5.terminal_info()
    account = rowdict(ai) if ai else {}
    terminal = rowdict(ti) if ti else {}
    now = datetime.now()
    start = now - timedelta(days=3650)
    deals_raw = mt5.history_deals_get(start, now) or []
    orders_raw = mt5.history_orders_get(start, now) or []
    positions_raw = mt5.positions_get(symbol='XAUUSD') or []
    mt5.shutdown()

    deals = [rowdict(d) for d in deals_raw]
    orders = [rowdict(o) for o in orders_raw]
    positions = [rowdict(p) for p in positions_raw]
    write_csv(out / 'mt5_history_deals_raw.csv', deals)
    write_csv(out / 'mt5_history_orders_raw.csv', orders)
    write_csv(out / 'mt5_live_positions.csv', positions)

    trade_deals = [d for d in deals if d.get('type') in (0, 1)]
    balance_deals = [d for d in deals if d.get('type') not in (0, 1)]
    by_pos: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for d in trade_deals:
        by_pos[d.get('position_id') or d.get('order') or d.get('ticket')].append(d)

    closed: List[Dict[str, Any]] = []
    open_groups: List[Dict[str, Any]] = []
    for pos_id, group in by_pos.items():
        group = sorted(group, key=lambda x: (x.get('time') or 0, x.get('ticket') or 0))
        entries = [g for g in group if g.get('entry') == 0]
        exits = [g for g in group if g.get('entry') in (1, 3)]
        net = sum(net_of_deal(g) for g in group)
        volume_in = sum(float(g.get('volume') or 0) for g in entries)
        volume_out = sum(float(g.get('volume') or 0) for g in exits)
        rec = {
            'account_login': account.get('login'),
            'account_server': account.get('server'),
            'position_id': pos_id,
            'open_time': group[0].get('time_iso'),
            'close_time': group[-1].get('time_iso') if exits else '',
            'symbol': group[0].get('symbol'),
            'side': group[0].get('type_name'),
            'volume_in': round(volume_in, 4),
            'volume_out': round(volume_out, 4),
            'open_price': entries[0].get('price') if entries else group[0].get('price'),
            'close_price': exits[-1].get('price') if exits else '',
            'gross_profit': round(sum(float(g.get('profit') or 0) for g in group), 2),
            'commission': round(sum(float(g.get('commission') or 0) for g in group), 2),
            'swap': round(sum(float(g.get('swap') or 0) for g in group), 2),
            'fee': round(sum(float(g.get('fee') or 0) for g in group), 2),
            'net': round(net, 2),
            'magic_open': entries[0].get('magic') if entries else group[0].get('magic'),
            'comment_open': entries[0].get('comment') if entries else group[0].get('comment'),
            'deal_tickets': ';'.join(str(g.get('ticket')) for g in group),
            'orders': ';'.join(str(g.get('order')) for g in group),
        }
        if exits and abs(volume_out - volume_in) < 1e-9:
            closed.append(rec)
        else:
            open_groups.append(rec)
    closed.sort(key=lambda r: r.get('close_time') or '')
    write_csv(out / 'closed_position_results.csv', closed)
    write_csv(out / 'open_position_groups_from_history.csv', open_groups)

    end_equity = float(account.get('equity') or account.get('balance') or 0.0)
    # Estimate start equity from current equity minus all closed trade net; deposits are separately recorded.
    start_equity = end_equity - sum(float(r['net']) for r in closed)
    curve = []
    eq = start_equity
    curve.append({'account_login': account.get('login'), 'time': '', 'event': 'estimated_start', 'equity': round(eq, 2), 'return_pct': 0.0, 'position_id': '', 'net': 0})
    for r in closed:
        eq += float(r['net'])
        curve.append({
            'account_login': account.get('login'),
            'time': r.get('close_time'),
            'event': 'closed_position',
            'equity': round(eq, 2),
            'return_pct': round((eq - start_equity) / start_equity * 100, 4) if start_equity else None,
            'position_id': r.get('position_id'),
            'net': r.get('net'),
        })
    write_csv(out / 'account_equity_curve_closed_trades.csv', curve)

    daily: Dict[str, Dict[str, Any]] = defaultdict(lambda: {'date': '', 'closed_positions': 0, 'net_pnl': 0.0})
    for r in closed:
        day = str(r.get('close_time') or '')[:10]
        daily[day]['date'] = day
        daily[day]['closed_positions'] += 1
        daily[day]['net_pnl'] += float(r['net'])
    daily_rows = []
    running = start_equity
    for day in sorted(daily):
        running += daily[day]['net_pnl']
        daily_rows.append({**daily[day], 'net_pnl': round(daily[day]['net_pnl'], 2), 'equity': round(running, 2), 'return_pct': round((running-start_equity)/start_equity*100, 4) if start_equity else None})
    write_csv(out / 'daily_account_returns.csv', daily_rows)

    log_events = extract_log_events()
    write_csv(out / 'strategy_log_order_signal_events.csv', log_events)

    summary = {
        'generated_at': datetime.now().isoformat(sep=' '),
        'output_dir': str(out),
        'account': {k: account.get(k) for k in ['login', 'server', 'balance', 'equity', 'currency', 'name', 'company']},
        'terminal_trade_allowed': terminal.get('trade_allowed'),
        'history_range_query': {'start': start.isoformat(sep=' '), 'end': now.isoformat(sep=' ')},
        'counts': {
            'raw_deals': len(deals),
            'raw_orders': len(orders),
            'trade_deals': len(trade_deals),
            'balance_nontrade_deals': len(balance_deals),
            'closed_positions': len(closed),
            'open_position_groups_from_history': len(open_groups),
            'live_positions': len(positions),
            'strategy_log_events': len(log_events),
        },
        'performance_current_account_closed_positions': summarize_closed(closed, start_equity, end_equity),
        'caveats': [
            'MT5 history API only returns history visible for the currently selected/login account in the terminal.',
            'Equity curve is reconstructed from closed trade net PnL; deposits/withdrawals and account switches require separate MT5 statements for perfect per-account accounting.',
            'strategy_log_order_signal_events.csv is an execution/log proxy, not authoritative realized PnL.',
        ],
        'files': {
            'raw_deals': str(out / 'mt5_history_deals_raw.csv'),
            'raw_orders': str(out / 'mt5_history_orders_raw.csv'),
            'closed_results': str(out / 'closed_position_results.csv'),
            'equity_curve': str(out / 'account_equity_curve_closed_trades.csv'),
            'daily_returns': str(out / 'daily_account_returns.csv'),
            'log_events': str(out / 'strategy_log_order_signal_events.csv'),
            'live_positions': str(out / 'mt5_live_positions.csv'),
        },
    }
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    md = [
        '# MT5 Live Trade Audit',
        '',
        f"Generated: {summary['generated_at']}",
        f"Account: {summary['account']}",
        '',
        '## Counts',
    ]
    for k, v in summary['counts'].items():
        md.append(f'- {k}: {v}')
    md += ['', '## Performance']
    for k, v in summary['performance_current_account_closed_positions'].items():
        md.append(f'- {k}: {v}')
    md += ['', '## Files']
    for k, v in summary['files'].items():
        md.append(f'- {k}: `{v}`')
    md += ['', '## Caveats'] + [f'- {c}' for c in summary['caveats']]
    (out / 'REPORT.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
