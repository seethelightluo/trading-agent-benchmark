Current date: 2029-11-15
Your previous output: ### Explored Factors

1. **Volume-confirmed 20-observation breakout**
   - Construction:
     \[
     F_{i,t}=\left(\frac{P_{i,t-20}}{P_{i,t}}-1\right)
     	imes \min\left(4,\frac{V_{i,t}}{\operatorname{median}(V_{i,t-60:t})}\right)
     \]
     lagged by one day and cross-sectionally residualized against lagged 20-day volatility.
   - Motivation: test whether medium-term returns confirmed by unusually high trading participation predict continued relative performance.
   - Universe: all 15 tradable benchmark instruments.
   - Data cutoff: 2029-10-31.
   - Observations: 3,248 rows; approximately 8.9 valid instruments per date.
   - Signal coverage: 38.23%.
   - Ten-day rank turnover: 0.6294.

### Validation Results

| Horizon | IC | ICIR | Hit ratio | Dates |
|---|---:|---:|---:|---:|
| 1 day | +0.03216 | +0.07769 | 53.87% | 1,149 |
| 5 days | +0.01018 | +0.02495 | 51.02% | 1,176 |
| 10 days | +0.01712 | +0.04200 | 52.41% | 1,015 |
| 20 days | +0.02923 | +0.06992 | 54.04% | 1,225 |

The candidate did not satisfy the binding admission contract at any horizon because no horizon simultaneously achieved:

- \(|IC| \ge 0.0070\)
- \(|ICIR| \ge 0.0840\)

Regime diagnostics showed material instability:

- 2026–2028: generally positive at the 5–10 day horizons, with 2028 H5 IC +0.07880 and ICIR +0.20163.
- 2029: H20 IC +0.07521 and ICIR +0.23733.
- Latest 120 dates: H20 IC +0.04132 and ICIR +0.11868.
- However, 2028 H20 performance was negative: IC −0.00420 and ICIR −0.01062.
- The factor was highly correlated with its underlying 20-day return component: pooled Spearman rho +0.84549, indicating substantial overlap with trend factors.

### Persistence Actions

- **No factor persisted.**
- The complete library maximum-correlation audit was not completed; therefore the candidate lacked the required correlation evidence in addition to failing the pooled admission gates.
- No existing factors were deprecated.
- Existing effective factors remain unchanged.

### Current Effective Factors

The active library remains unchanged, including the existing momentum, volatility, peer-dependence, tail-risk, and macro-resilience factors. No new factor was admitted in this cycle.

### Plans

- Explore volume signals after explicitly removing medium-term trend exposure, rather than merely volatility-residualizing them.
- Test event-style volume shocks, volume dry-ups, and abnormal participation reversals with lower correlation to existing trend factors.
- Prefer candidates with stable performance across 2028–2029 and complete maximum-correlation audits against every admitted factor before persistence.
...