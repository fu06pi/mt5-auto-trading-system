# Retired strategy: OCC / Open-Close Cross

Status: retired on 2026-06-15.

Reason: deterministic XAUUSD backtest showed persistent negative expectancy and excessive churn.

Evidence:
- Data: cached XAUUSD M5, 2024-12-18 to 2026-05-27.
- `OCC_M15_SMMA8_nonrepaint`: 13,262 trades, net -93,245.88, PF 0.599, closed DD 93.29%.
- `OCC_M15_DEMA8_nonrepaint`: 13,459 trades, net -95,145.98, PF 0.560.
- `OCC_M15_DEMA8_range_adx20`: 5,659 trades, net -73,593.32, PF 0.523.

Decision:
- Do not live-deploy.
- Do not include in all-sleeve activation scripts.
- If revisited, start from a new non-repainting research spec with strict trade-frequency and PF requirements.
