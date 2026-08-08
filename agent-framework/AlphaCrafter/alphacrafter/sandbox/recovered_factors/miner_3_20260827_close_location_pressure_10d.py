Current date: 2026-09-10
Your previous output: ## Research Cycle Summary — 2026-08-27

### Explored Factor
- **20-observation downside-excursion resilience**
  - **Construction:** for each asset, calculate the daily downside probe from prior close to daily low, scaled by the day’s high–low range, then take its negative 20-observation mean:
    \[
    -\operatorname{mean}_{20}\left(\max\left(0,\frac{Close_{t-1}-Low_t}{High_t-Low_t}\right)\right)
    \]
  - **Motivation:** assets with consistently shallow intraday downside excursions may demonstrate resilience and provide a distinct defensive/range-behavior signal.

### Validation Results
- Validation data through **2026-08-26**; factor validation date **2026-08-27**.
- Universe: all **15** tradable benchmark instruments.
- Signal-cell coverage: **58.75%**.
- Mean valid instruments per IC date: **8.81**.
- Mean cross-sectional rank turnover: **0.088670**.

| Forward horizon | Daily paper IC | Daily paper ICIR | IC hit ratio | IC dates |
|---|---:|---:|---:|---:|
| 1 observation | -0.001319 | -0.003746 | 51.37% | 1,355 |
| 5 observations | -0.005689 | -0.016634 | 48.67% | 1,013 |
| 10 observations | 0.017508 | 0.051425 | 50.40% | 1,002 |
| 20 observations | 0.048550 | 0.140909 | 55.01% | 1,336 |

The 20-observation return horizon has acceptable standalone IC and ICIR, but admission failed due to the mandatory library-diversification gate:

- Maximum absolute library Spearman correlation: **0.746027**
- Correlated factors: `miner_3_risk_adjusted_trend_20d` and `miner_1_ravmom_20obs`
- Required maximum: **strictly below 0.500000**

The five-observation horizon was unstable by regime:
- 2020: IC **-0.017508**, ICIR **-0.052507**
- 2021–22: IC **0.003567**, ICIR **0.010607**
- 2023–24: IC **-0.039889**, ICIR **-0.110443**
- 2025–26: IC **0.030090**, ICIR **0.091829**

This indicates the apparent long-horizon association is not sufficiently independent of the established trend/risk-adjusted momentum family.

### Persistence Actions
- **No factor persisted.**
- The candidate failed the binding maximum-library-correlation requirement despite passing the standalone 20-observation IC/ICIR thresholds.

### Current Effective Factors
The admitted library remains unchanged:

1. `miner_3_risk_adjusted_trend_20d`
2. `miner_3_relative_volume_participation_20d`
3. `miner_1_ravmom_20obs`
4. `miner_1_volnorm_reversal_5obs`
5. `miner_2_realized_volatility_20obs`
6. `miner_2_volscaled_reversal_1obs`

### Plans
- Avoid further straightforward downside-resilience scores that aggregate price excursions over medium windows, since they are strongly trend-correlated.
- Explore more orthogonal candidates based on:
  - volume-conditioned range expansion/contraction,
  - realized cross-asset dispersion or correlation-regime measures,
  - macro-conditioned signals using observation-only VIX, FX, and dollar series,
  - nonlinear tail-frequency measures separated from directional trend....