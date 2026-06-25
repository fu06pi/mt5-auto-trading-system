# Retired strategy: XAUUSD confluence pullback prototypes

Status: retired on 2026-06-15.

Reason: XAUUSD confluence pullback rules produced either no trades or strongly negative small-sample results.

Evidence:
- `research_xauusd_confluence_backtest.py` v1:
  - strict/no-vwap variants: 0 trades.
  - loose variants: 14–16 trades, roughly -8.08% to -8.77%, PF near 0.04–0.05 or 0.
- `research_xauusd_confluence_backtest_v2.py`:
  - 2–4 trades per variant.
  - all variants negative, roughly -8.03% to -11.34%.

Decision:
- Remove these XAUUSD prototypes.
- MNQ/NQ confluence research is separate and remains only a proxy/watchlist idea because it had a tiny positive sample, not enough for MT5 live development.
