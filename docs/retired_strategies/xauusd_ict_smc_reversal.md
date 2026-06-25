# Retired strategy: XAUUSD ICT/SMC reversal

Status: retired on 2026-06-15.

Reason: discretionary ICT/SMC idea did not survive deterministic rule encoding on cached broker data.

Evidence:
- Data: cached XAUUSD M5, 2025-05-27 to 2026-05-27.
- Best tested variant `ob_only`: 133 trades, net -3633.77, PF 0.786, max DD 4.79%.
- `baseline_buy_only`: 88 trades, net -5496.55, PF 0.579.
- `baseline_sell_only`: 111 trades, net -6126.91, PF 0.623.
- `baseline_both_fvg_ob_counter_session`: 188 trades, net -10239.20, PF 0.622.

Decision:
- Remove from active development and live-candidate lists.
- Do not port to active_plan.
- If ICT/SMC is revisited, it should be a fresh research project with separate validation and not this implementation.
