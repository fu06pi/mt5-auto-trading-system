#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

OUT_ROOT = Path('/home/chain4655/Documents/Projects/MT5/research/x_strategy_scan')
APP = 'my-app'
QUERIES = [
    'XAUUSD ORB strategy',
    'Gold futures ORB strategy',
    'XAUUSD liquidity sweep strategy',
    'XAUUSD London open strategy',
    'XAUUSD NY session strategy',
    'XAUUSD breakout strategy',
    'XAUUSD mean reversion strategy',
    'XAUUSD moving average resistance strategy',
    'Gold trading backtest strategy',
    'GC futures opening range breakout strategy',
    'XAUUSD ICT strategy liquidity sweep',
    'XAUUSD FVG strategy',
]
SCAM_PATTERNS = [
    r'whatsapp', r't\.me', r'telegram', r'free signals?', r'dm for', r'copy', r'profit zone',
    r'VIP', r'join', r'guaranteed', r'100%+', r'daily wins', r'be on the winning side',
]
GOOD_PATTERNS = {
    'ORB/opening-range': [r'\bORB\b', r'opening range', r'open range'],
    'liquidity-sweep': [r'liquidity sweep', r'sweep', r'stop hunt', r'raid'],
    'session-filter': [r'London', r'NY session', r'US Session', r'Asia session', r'New York'],
    'breakout': [r'breakout', r'break out', r'range break'],
    'mean-reversion': [r'mean reversion', r'buy dips', r'sell rallies', r'resistance', r'support'],
    'risk-defined': [r'stop', r'target', r'R:R', r'RR', r'risk'],
    'backtest/evidence': [r'backtest', r'win rate', r'profit factor', r'expectancy', r'journal'],
    'automation': [r'MT5', r'EA', r'Pine', r'TradingView', r'algorithm', r'algo'],
    'ICT/SMC/FVG': [r'ICT', r'SMC', r'FVG', r'fair value gap', r'order block'],
}


def run_query(query: str, n: int = 15) -> Dict[str, Any]:
    cmd = ['xurl', '--app', APP, 'search', query, '-n', str(n)]
    p = subprocess.run(cmd, text=True, capture_output=True, timeout=90)
    if p.returncode != 0:
        return {'query': query, 'error': p.stderr.strip() or p.stdout.strip()}
    try:
        data = json.loads(p.stdout)
    except json.JSONDecodeError as exc:
        return {'query': query, 'error': f'json decode: {exc}', 'raw': p.stdout[:500]}
    data['query'] = query
    return data


def metrics_score(m: Dict[str, Any]) -> float:
    likes = float(m.get('like_count') or 0)
    replies = float(m.get('reply_count') or 0)
    reposts = float(m.get('retweet_count') or 0)
    bookmarks = float(m.get('bookmark_count') or 0)
    impressions = float(m.get('impression_count') or 0)
    return likes * 1 + replies * 1.5 + reposts * 2 + bookmarks * 2 + min(impressions / 1000, 5)


def classify(text: str) -> tuple[List[str], int, List[str]]:
    hits: List[str] = []
    reasons: List[str] = []
    score = 0
    for name, pats in GOOD_PATTERNS.items():
        if any(re.search(p, text, flags=re.I) for p in pats):
            hits.append(name)
            score += 1
    if 'risk-defined' in hits:
        score += 2
        reasons.append('has entry/stop/target/risk language')
    if 'backtest/evidence' in hits:
        score += 2
        reasons.append('mentions backtest/evidence')
    if 'ORB/opening-range' in hits or 'liquidity-sweep' in hits or 'session-filter' in hits:
        score += 2
        reasons.append('matches session/intraday execution focus')
    if 'mean-reversion' in hits or 'breakout' in hits:
        score += 1
    scam = [p for p in SCAM_PATTERNS if re.search(p, text, flags=re.I)]
    if scam:
        score -= 3 + len(scam)
        reasons.append('promo/signal-room risk: ' + ', '.join(scam[:4]))
    if re.search(r'XAUUSD|GOLD|\bGC\b', text, re.I):
        score += 1
    if len(text) < 60:
        score -= 1
    return hits, score, reasons


def main() -> None:
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = OUT_ROOT / stamp
    out.mkdir(parents=True, exist_ok=True)
    raw_results = []
    for q in QUERIES:
        res = run_query(q, n=15)
        raw_results.append(res)
        time.sleep(1)
    (out / 'raw_xurl_search_results.json').write_text(json.dumps(raw_results, ensure_ascii=False, indent=2), encoding='utf-8')

    users: Dict[str, Dict[str, Any]] = {}
    posts: Dict[str, Dict[str, Any]] = {}
    for res in raw_results:
        for u in res.get('includes', {}).get('users', []) if isinstance(res, dict) else []:
            users[str(u.get('id'))] = u
        for p in res.get('data', []) if isinstance(res, dict) else []:
            pid = str(p.get('id'))
            rec = posts.setdefault(pid, dict(p))
            rec.setdefault('queries', []).append(res.get('query'))

    rows: List[Dict[str, Any]] = []
    for p in posts.values():
        text = p.get('text') or ''
        hits, score, reasons = classify(text)
        author = users.get(str(p.get('author_id')), {})
        m = p.get('public_metrics') or {}
        eng = metrics_score(m)
        total = score + min(eng, 8) * 0.5
        expanded_urls = []
        for url in (p.get('entities') or {}).get('urls', []) or []:
            expanded_urls.append(url.get('expanded_url') or url.get('url'))
        rows.append({
            'total_score': round(total, 2),
            'logic_score': score,
            'engagement_score': round(eng, 2),
            'created_at': p.get('created_at'),
            'author_username': author.get('username'),
            'author_name': author.get('name'),
            'verified': author.get('verified'),
            'tweet_id': p.get('id'),
            'url': f"https://x.com/{author.get('username')}/status/{p.get('id')}" if author.get('username') else f"https://x.com/i/status/{p.get('id')}",
            'strategy_tags': ';'.join(hits),
            'reason': '; '.join(reasons),
            'likes': m.get('like_count'),
            'replies': m.get('reply_count'),
            'retweets': m.get('retweet_count'),
            'bookmarks': m.get('bookmark_count'),
            'impressions': m.get('impression_count'),
            'queries': '; '.join(p.get('queries', [])),
            'expanded_urls': ';'.join([u for u in expanded_urls if u]),
            'text': text.replace('\n', ' '),
        })
    rows.sort(key=lambda r: (float(r['total_score']), float(r['engagement_score'])), reverse=True)

    csv_path = out / 'ranked_x_strategy_candidates.csv'
    fields = list(rows[0].keys()) if rows else []
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # Strategy synthesis: manually group tags into testable candidates.
    top = rows[:25]
    tag_counts: Dict[str, int] = {}
    for r in rows:
        for t in str(r['strategy_tags']).split(';'):
            if t:
                tag_counts[t] = tag_counts.get(t, 0) + 1
    md = [
        '# X Strategy Candidate Scan — XAUUSD/Gold',
        '',
        f'- Generated: {datetime.now().isoformat(sep=" ", timespec="seconds")}',
        f'- Queries: {len(QUERIES)}',
        f'- Unique posts collected: {len(rows)}',
        f'- Output CSV: `{csv_path}`',
        '',
        '## Tag counts',
    ]
    for k, v in sorted(tag_counts.items(), key=lambda kv: kv[1], reverse=True):
        md.append(f'- {k}: {v}')
    md += ['', '## Top candidates / ideas']
    for i, r in enumerate(top, 1):
        md += [
            f'### {i}. score={r["total_score"]} @{r["author_username"]}',
            f'- URL: {r["url"]}',
            f'- Tags: {r["strategy_tags"] or "none"}',
            f'- Reason: {r["reason"] or "n/a"}',
            f'- Metrics: likes={r["likes"]}, replies={r["replies"]}, reposts={r["retweets"]}, bookmarks={r["bookmarks"]}, impressions={r["impressions"]}',
            f'- Text: {r["text"][:600]}',
            '',
        ]
    md += [
        '## Extracted strategy hypotheses for our MT5/XAUUSD target',
        '',
        '1. **US/NY session ORB short/long**: opening range breakout on GC/XAUUSD with hard stop and 1.5R–2R target; test M5/M15, session window filter, spread cap, one trade per session.',
        '2. **Sell-rallies / buy-dips around intraday support-resistance**: detect MA/resistance/support touch after directional impulse; test as mean-reversion or trend-continuation variant with ATR stop.',
        '3. **Liquidity sweep reversal**: after sweep of Asian/London high-low, enter on reclaim/confirmation; test London open and NY open separately.',
        '4. **ICT/FVG/order-block continuation**: many posts mention SMC/ICT language, but most lack exact rules; only use if converted into objective candle/FVG rules.',
        '',
        '## Filters used',
        '- Positive: ORB, liquidity sweep, session filter, breakout, mean reversion, risk-defined entry/SL/TP, backtest/evidence, MT5/Pine automation mentions.',
        '- Negative: Telegram/WhatsApp/free-signal/DM/VIP promo language.',
        '',
        '## Caveat',
        'Most XAUUSD Twitter results are signal-room promotions. Treat these as idea seeds only; do not trade without converting to objective rules and backtesting on our XAUUSD data.',
    ]
    report_path = out / 'REPORT.md'
    report_path.write_text('\n'.join(md) + '\n', encoding='utf-8')
    print(json.dumps({'out_dir': str(out), 'csv': str(csv_path), 'report': str(report_path), 'unique_posts': len(rows), 'top': rows[:8]}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
