# 001: False-breakout sharp reversal signal

## Question
Can the current XAUUSD M5 trend-following framework exploit fake breakouts by entering the reversal direction after a wick pierces a recent range extreme but closes back inside?

## Signal definition tested

- **Upthrust SELL**: previous closed M5 bar breaks above the prior range high, has a large upper wick, then closes back under the range high.
- **Spring BUY**: previous closed M5 bar breaks below the prior range low, has a large lower wick, then closes back above the range low.
- Uses the existing trend-following H1 proxy filter:
  - `with_htf`: only SELL in H1 BEAR / BUY in H1 BULL.
  - `counter_htf`: only opposite H1 trend.
  - `any`: no HTF direction filter.
- Test data: `/home/chain4655/Documents/backtest_reports/xauusd_5y_weekly_reset/data/XAUUSD_M5_5y_mt5.csv`
- Simulator: next-bar open entry, conservative same-bar SL-first handling, R-multiple output.

## Key result

Best usable edge is **one-direction SELL-only upthrust in BEAR HTF**, not symmetric BUY+SELL.

- Variant: `strict_with_htf_3r`, filtered to SELL-only
- Trades: 533
- Win rate: 30.39%
- Total: +86.80R
- Avg/trade: +0.163R
- Max DD: -18.83R
- 2025: 315 trades, +44.53R, +0.141R/trade
- 2026: 218 trades, +42.27R, +0.194R/trade

The BUY side under the same logic is negative:

- `strict_with_htf_3r` BUY-only: 1008 trades, -42.17R, -0.042R/trade

## Full variant summary

See `results/summary.csv`.

Top full-symmetric variants:

- `strict_with_htf_2r`: 1584 trades, +58.32R, avg +0.037R, max DD -49.09R
- `strict_with_htf_3r`: 1541 trades, +44.63R, avg +0.029R, max DD -83.53R

The symmetric signal is too diluted. SELL-only carries the edge.

## Verdict: PARTIAL / ACTIONABLE

### What worked
- Fake upside breakout reversal works when aligned with existing H1 BEAR trend.
- The signal is additive to the trend-following logic: it catches failed upside pushes inside a bearish regime.
- SELL-only is stable across 2025 and 2026 sample periods.

### What did not work
- BUY-side spring signal is negative in this dataset.
- Counter-HTF reversal is negative; do **not** use this as generic mean reversion.
- Stricter wick filters reduce trade count but did not improve edge.

### Recommendation for production build

Add a disabled-by-default overlay to `mt5_xauusd_trend_strategy.py`:

- Flag: `--enable-false-breakout-reversal`
- Direction: default `SELL_ONLY`
- Conditions:
  - `compensated_htf_signal == BEAR`
  - previous closed M5 high pierces prior 20-bar high by >= 0.15 ATR
  - close returns below prior high by >= 0.05 ATR
  - upper wick ratio >= 0.45
  - entry SELL on next loop/bar if no position and normal risk gates pass
- Initial TP should use 3R for this overlay if separate TP routing is implemented; otherwise current `reward_multiple=2.5` is acceptable but less optimal than 3R.

Do not enable live until we either:
1. add this overlay with explicit logging and separate counters, then
2. run a dry-run/live-shadow period or a more exact event-driven backtest using current execution rules.
