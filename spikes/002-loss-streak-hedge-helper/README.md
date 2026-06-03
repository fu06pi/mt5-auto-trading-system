# 002: Trend weak-regime hedge helper

## Question

Can we build a research-only auxiliary hedge strategy that activates during weak periods of the current XAUUSD trend strategy — primarily range/chop regimes where trend-following expectancy deteriorates — and reduces drawdown without destroying expectancy?

## Hypothesis

The current trend strategy has weak regimes that can be detected by market structure and trend-quality metrics. Consecutive realized losses are only one diagnostic symptom, not the core trigger. A better helper sleeve should detect chop/range conditions and either hedge, fade false continuations, or pause trend exposure during those regimes.

This is not intended as a production implementation yet. It is a spike to test whether the concept has statistical value.

## Approach

- Data: cached broker M5 XAUUSD bars.
- Data range: 2024-12-18 02:45 to 2026-05-27 13:50.
- Bars: 100,470 M5 bars.
- Initial equity: 100,000.
- Baseline: current trend logic from `research_xauusd_trend_plus_complement_backtest.py`.
- Hedge helper:
  - independent sleeve;
  - primary target is trend weak regime / range chop, not loss streak itself;
  - loss streak is treated as a backward-looking diagnostic label for weak periods;
  - future trigger candidates should use objective chop metrics such as ADX compression, EMA slope decay, ATR contraction, range efficiency, alternating BUY/SELL signals, and failed breakout frequency;
  - has separate risk multiplier, RR, and max-hold parameters;
  - shares account equity only in simulation.

The first spike tested loss-streak proxies because they are easy to reconstruct from realized trend trades. This is **not** the final intended trigger.

1. **Naive loss-streak hedge** — after N losses, hedge opposite of any next trend signal.
2. **Same-side loss-streak hedge** — after N losses, only hedge if the next signal is the same side as the side that just clustered losses. This targets repeated false continuation rather than every post-loss signal.

## Run

Loss-streak proxy spike:

```bash
cd /home/chain4655/Documents/Projects/MT5
python3 spikes/002-loss-streak-hedge-helper/loss_streak_hedge_spike.py
```

Direct chop-regime spike:

```bash
cd /home/chain4655/Documents/Projects/MT5
python3 spikes/002-loss-streak-hedge-helper/chop_regime_hedge_spike.py
```

Outputs:

- `results/loss_streak_hedge_grid.csv`
- `results/loss_streak_hedge_diagnostics.json`
- `results/best_hedge_trades.csv`
- `results/best_combined_trades.csv`
- `results/chop_regime_hedge_grid.csv`
- `results/chop_regime_hedge_diagnostics.json`
- `results/best_chop_helper_trades.csv`
- `results/best_chop_combined_trades.csv`

## Key observation from losing streak cases

Current trend losing streaks are often mixed BUY/SELL chop, not clean one-direction failure. This supports the user's correction: the hedge target should be **trend weak regime /震盪**, not the fact that losses happened consecutively.

Examples from the current trend trade log:

- Max losing streak length: 15 trades.
- 2025-W26: 15 losses, net -6,303.46, BUY 7 / SELL 8.
- 2025-W33: 15 losses, net -6,568.94, BUY 7 / SELL 8.
- 2026-W20: 14 losses, net -8,097.26, BUY 8 / SELL 6.

Implication: a naive opposite hedge after losses tends to overtrade chop and becomes a second losing strategy.

## Best candidate result

Candidate: `same_side_ls4_r0.5_rr2.0_act12`

Rules:

- Trigger after 4 consecutive trend losses.
- Only hedge when the next trend signal is the same side as the side that just clustered losses.
- Hedge side = opposite of that trend signal.
- Hedge risk multiplier = 0.5 of baseline trend risk.
- Hedge RR = 2.0.
- Hedge max hold = 48 M5 bars.
- Activation window = 12 M5 bars.

Result:

- Hedge trades: 204.
- Hedge standalone return: +1.36%.
- Hedge PF: 1.053.
- Hedge max DD: -2.25%.
- Combined return: +61.12%.
- Combined PF: 1.074.
- Combined max DD: -12.17%.
- Return delta vs trend simulation: +1.35%.
- DD improvement vs trend simulation: +1.10%.

## Rejected result

Naive loss-streak hedge was rejected.

Example: `naive_ls2_r0.25_rr1.0_act48`

- Hedge trades: 2,393.
- Hedge standalone return: -20.30%.
- Hedge PF: 0.831.
- Combined return: +32.35%.

This reduces some weekly pain but destroys expectancy. It is not a viable hedge strategy.

## Verdict: PARTIAL

The broad idea is **partially validated**:

- Consecutive-loss regions exist and are measurable.
- A targeted same-side loss-streak hedge can produce mildly positive expectancy.
- The effect is too small for the user's 4% weekly target.
- Naive opposite hedging is invalidated.

## Direct chop-regime spike result

Script: `chop_regime_hedge_spike.py`

Detector features:

- ADX compression.
- Range efficiency.
- ATR contraction.
- SMA20/SMA60 flatness versus ATR.
- BUY/SELL alternation.
- Weak primary trend score.

Baseline trend-only result on the same 100k cached M5 dataset:

- Trades: 2,632.
- Return: +55.11%.
- PF: 1.072.
- Max DD: -13.10%.

Best defensive result:

- Candidate: `chop_scale_balanced_r0.50`.
- Action: scale trend risk to 0.5x during detected chop.
- Chop bars: 49,924.
- Return: +39.90%.
- PF: 1.068.
- Max DD: -10.14%.
- DD improvement vs baseline: +2.96%.
- Return delta vs baseline: -15.21%.

Best return + DD result remains:

- Candidate: `chop_pause_strict`.
- Action: pause trend entries during strictly detected chop.
- Chop points required: 3 of 6.
- Chop bars: 34,288.
- Trades: 2,503.
- Return: +58.97%.
- PF: 1.081.
- Max DD: -11.47%.
- DD improvement vs baseline: +1.63%.
- Return delta vs baseline: +3.86%.

Follow-up parameter test:

- `min_chop_points=4` became too strict: DD improvement collapsed to about +0.02% and return fell -3.52%.
- 0.5x scaling is better for DD but sacrifices too much return.
- 0.25x scaling with strict p4 keeps return close to baseline but barely improves DD.

Session-aware improvement test:

- Candidate: `chop_session_gate_balanced_r0.50`.
- Rule: Asia chop = full pause; non-Asia chop = 0.5x scale.
- Return: +59.46%.
- PF: 1.089.
- Max DD: -11.83%.
- DD improvement vs baseline: +1.27%.
- Return delta vs baseline: +4.35%.
- Interpretation: this currently gives the best return/PF while still improving DD, but DD improvement is slightly weaker than `chop_pause_strict`.

Best pure DD reducer:

- Candidate: `chop_session_gate_strict_r0.25`.
- Return: +44.58%.
- Max DD: -9.66%.
- DD improvement vs baseline: +3.44%.
- Return delta vs baseline: -10.53%.
- Interpretation: good for prop-style DD control, but too much return drag if the objective is growth.

Trade-path diagnostic for `chop_pause_strict`:

- Baseline-only trades: 1,223 trades, +38,808.87 net.
- Pause-only replacement trades: 1,094 trades, +43,380.89 net.
- Delta from path replacement: +4,572.03.
- Interpretation: the edge is not simply “blocked bad trades”; pausing chop changes cooldown/entry path and produces a better replacement trade set.
- Asia-session baseline-only trades were negative (-4,980.50), while late-US / overlap / London-pre-US baseline-only trades were positive. This suggests future improvement should apply stricter chop pause in Asia, and use lighter scaling rather than full pause in stronger sessions.

Rejected in this run:

- Direct `hedge_opposite` helpers reduced or destroyed return.
- Direct `fade_breakout` helpers were negative standalone and worsened portfolio results.

Interpretation: the first direct chop-regime test supports **risk gating / pause** more than opening a new opposite hedge sleeve. If we still want a true hedge sleeve, it needs a stricter entry model than simply opposite-trading trend signals during chop.

## Recommendation for real build

Do not deploy live yet. The next research step should refine the direct **trend weak-regime / chop detector** and convert the best defensive result into a production-style gating candidate:

1. Convert `chop_pause_strict` into a session-aware gate:
   - Asia: full pause under strict chop.
   - London/US overlap/late-US: test 0.5x scaling or no action, because blocked baseline trades there were net-positive.
2. Verify week-by-week which drawdown clusters are improved, especially 2025-W26, 2025-W31, 2025-W33, 2025-W43, 2026-W07, and 2026-W19.
3. Compare pause vs 0.25x / 0.5x scaling using fixed baseline deltas, not variant-local trend metrics.
4. If a true hedge sleeve is still desired, require a more selective trigger than `opposite primary signal`; current direct hedge/fade variants are negative.
5. Keep loss streak as an evaluation/diagnostic feature, not as the main activation trigger.
6. Add a total exposure cap across trend + any future hedge.
7. Test whether strict/session-aware chop gating combines better with the SELL-only false-breakout complement sleeve.

## Production-style conservative gate backtest

Integrated into `research_xauusd_trend_plus_complement_backtest.py` as `chop_gate="conservative_session"`.

Rule:

- Strict chop detector: ADX <= 18, efficiency <= 0.18, ATR ratio <= 0.85, SMA slope <= 1.0 ATR, alternation >= 0.55, or weak score <= 0.65; trigger at 3 of 6 points.
- Asia + chop: full pause trend entries.
- Non-Asia + chop: scale trend risk and max lots to 0.25x.
- Complement SELL-only false-breakout sleeve is not directly reduced by this gate.

Long cached backtest, 100k initial equity, 2024-12-18 to 2026-05-27:

- Current trend baseline: return +58.59%, PF 1.075, DD -13.10%.
- Current trend + conservative chop: return +47.64%, PF 1.073, DD -9.83%.
- Overlay trend+complement baseline: return +54.17%, PF 1.071, DD -13.10%.
- Overlay trend+complement + conservative chop: return +49.17%, PF 1.075, DD -9.73%.
- Parallel independent sleeves baseline: return +84.07%, PF 1.094, DD -14.04%.
- Parallel independent sleeves + conservative chop: return +71.05%, PF 1.095, DD -10.38%.
- Standalone complement baseline: return +15.85%, PF 1.438, DD -3.17%.

Interpretation:

- Conservative session-aware chop gate improves DD materially across trend-only, overlay, and independent-sleeve portfolio modes.
- Best DD among current+complement variants is overlay + conservative chop at -9.73%.
- Best total return with materially improved DD is parallel independent sleeves + conservative chop: +71.05% return with DD reduced from -14.04% to -10.38%.
