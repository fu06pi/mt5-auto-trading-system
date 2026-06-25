# Strategy review — 2026-06-15

Context: after the morning postmortem on last week's MT5 losses, the review standard is stricter than simple positive backtest PnL. A candidate must survive direction quality, product-specific SL/TP geometry, risk-gate realism, and operational simplicity. Recent failure modes were: trend-tail chasing, mirrored XAU parameters on US100, delayed exit/reversal protection, manual/foreign positions masking strategy PnL, and too many sleeves increasing account-level complexity.

## Decision rubric

- **Keep / develop**: has repeatable positive expectancy or clear portfolio role, with enough trades and manageable DD.
- **Watchlist / research-only**: has some edge but sample is thin, PF is weak, or live-risk mechanics are not mature.
- **Retire**: negative expectancy, no-trade/too-thin signal, duplicate idea with worse evidence, or violates the current operational direction.

## Strategy decisions

### Keep / develop

- **XAUUSD trend main (`mt5_xauusd_trend_strategy.py`)**
  - Role: primary sleeve only; should remain single-sleeve-first for funded/live readiness.
  - Evidence: recent active work and current challenge deployment focus.
  - Required development: shorter product-specific SL/TP, strict raw HTF gate, stale-HTF reversal protection, loss/DD gates before order send, no blind US100 mirroring.

- **Meta regime switch (`mt5_meta_regime_switch_strategy.py`)**
  - Evidence: last-3m comparison: 189 trades, net +2051.23, PF 1.543, max DD -3.31%.
  - Development status: promising research/live candidate, but should be tested as a separate sleeve only after portfolio overlap and DD-stop simulation.

- **Momentum Surfer (`mt5_xauusd_momentum_surfer_strategy.py`)**
  - Evidence: last-3m comparison: 115 trades, net +1108.27, PF 1.234, max DD -5.64%; prior weekly notes warn some trailing variants damage PF.
  - Development status: watch as high-beta momentum sleeve; do not promote without weekly robustness and cost-inclusive PF.

- **IFVG Sniper (`mt5_ifvg_sniper_strategy.py`)**
  - Evidence: cached XAU M5 2024-12-18 to 2026-05-27: best variants positive, e.g. `ifvg_line_balanced_1r` 1793 trades, net +3773.71, PF 2.058, low reported DD.
  - Caveat: very high trade count and simplified execution assumptions; must receive live-realism audit before activation.

- **BBRSI / BB+RSI ranging (`mt5_xauusd_bb_rsi_ranging_strategy.py`)**
  - Evidence: feature-exit research: `adx20_all_signal_flip_exit` 134 trades, +27.65R; expanded ranging comparison best BBRSI around +17.94R, PF 1.26.
  - Caveat: not suitable as broad XAU ranging sleeve without regime-boundary hardening; better as small, disabled-by-default candidate.

### Watchlist / research-only

- **Bollinger edge squeeze (`mt5_bollinger_edge_squeeze_strategy.py`)**
  - Evidence: last-3m comparison: 152 trades, net +909.46, PF 1.387, max DD -3.18%.
  - Development status: has edge but overlaps with momentum/regime logic; keep as research candidate, not live by default.

- **Doomsday V4 / Doomsday (`mt5_doomsday*_strategy.py`)**
  - Evidence: last-3m V4 active sprint: 149 trades, net +1038.75, PF 1.236, max DD -6.07%, but DD guard breached in the simulator output.
  - Decision: keep only as high-risk simulation/challenge research, not funded/live. It conflicts with current safety-first funded readiness unless explicitly approved as pass/fail demo.

- **Tide-wave grid (`mt5_xauusd_tide_wave_grid_strategy.py`)**
  - Evidence mixed: last-3m selective M15 +143.10, PF 1.12, but M5 defensive -1078.39 and sprint companion -553.95. Grid logic has high win rate but weak PF and can hide tail risk.
  - Decision: freeze; do not live-deploy. Keep only if future work models basket floating DD and tail risk more realistically.

- **Asia/London breakout EP1 (`mt5_xauusd_ep1_strategy.py`)**
  - Evidence: last-3m only 1 trade (+9.58); older stability sample was positive but small.
  - Decision: keep as low-priority session-breakout candidate; not enough sample for live or deletion.

- **MNQ/NQ confluence pullback (`research_mnq_confluence_backtest.py`)**
  - Evidence: proxy result positive but only 5–10 trades in best variants.
  - Decision: keep as non-MT5/proxy research note; needs broker data before any serious development.

### Retired / removed from active development

- **OCC / Open-Close Cross (`mt5_occ_open_close_cross_strategy.py`, `research_occ_xauusd_backtest.py`)**
  - Evidence: direct run on cached XAU M5 2024-12-18 to 2026-05-27 was strongly negative. Examples: SMMA8 non-repaint 13,262 trades, net -93,245.88, PF 0.599, closed DD 93.29%; DEMA8 range ADX20 net -73,593.32, PF 0.523.
  - Decision: retire. It adds high churn and weak direction quality; directly conflicts with the postmortem focus on fewer, higher-quality entries.

- **XAUUSD ICT/SMC reversal (`mt5_xauusd_ict_smc_reversal_strategy.py`, `research_xauusd_ict_smc_reversal_backtest.py`)**
  - Evidence: one-year cached backtest all variants negative. Best variant `ob_only`: 133 trades, net -3633.77, PF 0.786; baseline both FVG/OB counter-session: 188 trades, net -10239.20, PF 0.622.
  - Decision: retire. The discretionary concept did not survive deterministic rule encoding on this data.

- **XAUUSD confluence pullback prototypes (`research_xauusd_confluence_backtest.py`, `research_xauusd_confluence_backtest_v2.py`)**
  - Evidence: v1 strict/no-vwap variants had zero trades; loose variants lost about -8% to -8.8%. v2 variants had only 2–4 trades and all negative, roughly -8% to -11.3%.
  - Decision: retire for XAUUSD. If confluence pullback is revisited, use a new broker-data spec rather than these prototypes.

## Operational conclusion

The near-term development stack should be narrow:

1. XAUUSD trend main as the only live-ready core after risk/exit fixes.
2. Meta-regime / Momentum / IFVG / BBRSI as separate research candidates, not bundled live sleeves.
3. Doomsday only for explicit demo/challenge pass-fail experiments.
4. Retired prototypes remain documented here; source files were removed locally after this retirement note was committed/pushed.
