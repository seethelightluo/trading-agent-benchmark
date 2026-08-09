Current date: 2026-08-27
Your previous output: ## Research Cycle Summary — 2026-08-13

### Explored Factor
- **Five-observation close-location reversal**
  - **Construction:** negative rolling mean of close location within the daily high–low range:
    \[
    -\frac{1}{5}\sum_{i=0}^{4}\frac{Close_{t-i}-Low_{t-i}}{High_{t-i}-Low_{t-i}}
    \]
  - **Motivation:** assets repeatedly closing near their daily lows may exhibit short-horizon rebound behavior. This uses intraday range information rather than only close-to-close returns.
  - Calculated on each asset’s native observation calendar through **2026-08-12**.

### Validation Results
- Universe: all **15** tradable cross-asset instruments.
- Mean valid instruments per IC date: **11.68**.
- IC observations: **1,701** at the primary one-observation horizon.
- Signal valid-cell coverage: **59.06%**.
- Mean cross-sectional rank turnover: **0.3397**.

| Forward horizon | Daily paper IC | Daily paper ICIR | IC hit ratio | IC dates |
|---|---:|---:|---:|---:|
| 1 observation | **0.040883** | **0.116870** | 55.26% | 1,701 |
| 5 observations | 0.007819 | 0.022891 | 50.09% | 1,697 |
| 10 observations | 0.001519 | 0.004472 | 49.70% | 1,692 |
| 20 observations | -0.004479 | -0.013214 | 49.41% | 1,682 |

The factor cleared the standalone one-day admission thresholds:
- \(|IC| = 0.040883 \ge 0.0070\)
- \(|ICIR| = 0.116870 \ge 0.0840\)

However, it failed the mandatory diversification gate:
- Maximum absolute library Spearman correlation: **0.725725**
- Correlation was with `miner_1_volnorm_reversal_5obs`.
- This materially exceeds the required maximum of **0.5000**.

Regime results for the 5-observation horizon, included as a medium-horizon robustness check, were also weak and uneven:
- 2020: IC **-0.027635**, ICIR **-0.074063**
- 2021–22: IC **0.025545**, ICIR **0.077893**
- 2023–24: IC **0.007207**, ICIR **0.021164**
- 2025–26: IC **0.008460**, ICIR **0.024989**

### Persistence Actions
- **No factor persisted.**
- Although its one-observation reversal signal was statistically above the standalone threshold, it is highly redundant with the existing five-observation volatility-normalized reversal factor and lacks stable medium-horizon efficacy.

### Current Effective Factors
The admitted library remains unchanged, with the most recently validated/addmitted short-horizon factor dated 2026-07-30:

1. `miner_3_risk_adjusted_trend_20d`
2. `miner_3_relative_volume_participation_20d`
3. `miner_1_ravmom_20obs`
4. `miner_1_volnorm_reversal_5obs`
5. `miner_2_realized_volatility_20obs`
6. `miner_2_volscaled_reversal_1obs`

### Plans
- Avoid further simple close-location reversal variants, as this result confirms substantial overlap with the active reversal family.
- Explore lower-correlation candidates based on:
  - downside range asymmetry and tail-frequency measures,
  - volume-conditioned range or breakout participation,
  - cross-asset dispersion and correlation-regime signals,
  - macro-conditioned defensive asset selection using observation-only macro series as conditioning variables....