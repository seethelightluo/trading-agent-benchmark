Current date: 2026-09-10
Your previous output: ## Research Cycle — 2026-08-27

### Explored Factors
- **Trend-orthogonal downside-energy residual, 20 observations**
  - Base signal: fraction of rolling squared-return energy arising on negative-return days:
    \[
    \frac{\sum_{20}\min(r_t,0)^2}{\sum_{20}r_t^2}
    \]
  - Orthogonalization: on each date, cross-sectionally regressed the base signal on the existing 20-day risk-adjusted trend factor and retained the OLS residual.
  - Motivation: preserve downside-tail asymmetry information while explicitly removing the very high overlap previously found with risk-adjusted trend.

### Validation Results
- Validation data through: **2026-08-26** (the completed day available at the current cursor)
- Full history: **2020-01-01 to 2026-08-26**
- Universe: **15 tradable cross-asset instruments**
- Valid cross-sectional IC dates: **1,675–1,694**, with at least 8 valid instruments.
- Mean IC-observation coverage: **96.9%–97.0%**
- Overall signal-cell coverage: **67.9%**, driven by the 20-observation/minimum-15-history requirement.
- Rank turnover: **0.1192**

| Forward horizon | Mean daily Spearman IC | ICIR | Hit ratio | Admission result |
|---|---:|---:|---:|---|
| 1 day | -0.013413 | -0.044345 | 47.6% | Fail |
| 5 days | -0.020385 | -0.067005 | 45.9% | Fail |
| 10 days | -0.015255 | -0.049939 | 46.2% | Fail |
| 20 days | -0.005372 | -0.018721 | 48.7% | Fail |

The factor passed the absolute IC threshold at 1-, 5-, and 10-day horizons, but **did not pass the binding absolute ICIR threshold of 0.0840** at any same horizon. Its best absolute ICIR was only **0.067005** at five days.

Regime results also show material instability:
- **2020:** weak / near-flat predictive behavior.
- **2021–22:** modestly negative.
- **2023–24:** positive at longer horizons, including 20-day IC of **0.044335** and ICIR of **0.155465**.
- **2025–26:** strongly negative, especially at 10- and 20-day horizons, with 20-day IC **-0.088048** and ICIR **-0.312665**.

### Library Correlation Screening
The residualization worked as intended for distinctness:

| Existing admitted factor | Spearman correlation |
|---|---:|
| `miner_3_risk_adjusted_trend_20d` | -0.037209 |
| `miner_3_relative_volume_participation_20d` | 0.010279 |
| `miner_1_volnorm_reversal_5obs` | -0.054795 |
| `miner_2_realized_volatility_20obs` | -0.059946 |

- `validation.metrics.max_abs_library_correlation`: **0.059946**
- This is well below the mandatory **0.5000** distinctness ceiling.

### Persistence Actions
- **No factor persisted.**
- Although it was highly distinct from all four admitted factors, it failed the required same-horizon ICIR gate and displayed significant regime drift.

### Current Effective Factors
No changes to the active library. The latest effective set remains:
1. `miner_3_risk_adjusted_trend_20d`
2. `miner_3_relative_volume_participation_20d`
3. `miner_1_volnorm_reversal_5obs`
4. `miner_2_realized_volatility_20obs`

### Plans
- Investigate a **volume-conditioned residual tail-asymmetry** signal, retaining the orthogonalization step but only activating tail information when volume participation is elevated.
- Revalidate the active volatility-normalized reversal and realized-volatility factors against the updated cursor, with emphasis on their narrow ICIR margins and recent-regime stability.
...