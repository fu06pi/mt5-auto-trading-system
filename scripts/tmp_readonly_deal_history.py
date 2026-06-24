
from __future__ import annotations
import json
from datetime import datetime, timedelta, timezone
from pymt5linux import MetaTrader5
HOST='127.0.0.1'; PORT=18812
mt5=MetaTrader5(host=HOST, port=PORT)
report={'initialize_ok': False}
if not mt5.initialize(path=r'C:\Program Files\MetaTrader 5\terminal64.exe'):
    report['last_error']=mt5.last_error(); print(json.dumps(report, default=str, indent=2)); raise SystemExit(2)
try:
    now=datetime.now(timezone.utc)
    start=now-timedelta(hours=12)
    deals=mt5.history_deals_get(start, now) or []
    rows=[]
    for d in deals:
        x=d._asdict() if hasattr(d,'_asdict') else dict(d)
        profit=float(x.get('profit',0) or 0)
        # include all nonzero-ish and all known magics
        rows.append({
            'time': datetime.fromtimestamp(int(x.get('time',0)), tz=timezone.utc).isoformat(),
            'ticket': x.get('ticket'), 'order': x.get('order'), 'position_id': x.get('position_id'),
            'symbol': x.get('symbol'), 'type': x.get('type'), 'entry': x.get('entry'),
            'volume': x.get('volume'), 'price': x.get('price'), 'profit': profit,
            'commission': x.get('commission'), 'swap': x.get('swap'), 'fee': x.get('fee'),
            'magic': x.get('magic'), 'comment': x.get('comment'), 'reason': x.get('reason')
        })
    # sort and print compact all rows with profit !=0 or known symbols
    rows.sort(key=lambda r: r['time'])
    report={'initialize_ok': True, 'from': start.isoformat(), 'to': now.isoformat(), 'count': len(rows), 'rows': rows[-80:]}
    print(json.dumps(report, default=str, indent=2))
finally:
    mt5.shutdown()
