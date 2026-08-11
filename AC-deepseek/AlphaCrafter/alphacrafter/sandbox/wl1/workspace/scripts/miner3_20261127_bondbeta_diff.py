"""miner_3 candidate: bond_beta_diff_60 (US10Y beta minus CN10Y beta, rolling 60d).

Motivation: cycle-8 screen flagged bond_beta_diff_60 with very low library
correlation (rho_max=0.017) and icir=-0.105. Macro cross-asset factor: assets
whose returns load more on US10Y than CN10Y (rate-sensitive) tend to
underperform. Re-validate on extended window through 2026-11-26.

Construction: for each asset i, rolling 60d beta of asset returns vs US10Y
returns minus rolling 60d beta vs CN10Y returns (min 30 obs). Rank IC is
invariant to the per-asset z-score used in the original screen, so the raw
beta-difference panel is validated here.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner3_20261008_lib import load_close_panel, run_validation, make_artifact

close = load_close_panel()
ret = close.pct_change()
lr = np.log(close / close.shift(1))

def roll_beta(x, m, win=60, minp=30):
    out = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
    for s in x.columns:
        cov = x[s].rolling(win, min_periods=minp).cov(m)
        var = m.rolling(win, min_periods=minp).var()
        out[s] = cov / (var + 1e-12)
    return out

beta_us10y = roll_beta(lr, lr["US10Y"])
beta_cn10y = roll_beta(lr, lr["CN10Y"])
factor = beta_us10y - beta_cn10y

print(f"panel dates={close.shape[0]} assets={close.shape[1]} "
      f"range={close.index.min().date()}..{close.index.max().date()}")
print(f"beta diff coverage: valid cells={factor.notna().sum().sum()} "
      f"({factor.notna().mean().mean():.3f} of asset-days)")

notes = ("US10Y-beta minus CN10Y-beta (rolling 60d, min 30 obs) cross-asset "
         "rate-sensitivity differential. Validated on 15-asset tradable universe "
         "through 2026-11-26. Regimes: 2020 COVID crash, 2021 bull, 2022 bear, "
         "2023-24 AI rally, 2025-26 crypto/commodity cycles, 2026 rate regime. "
         "Direction: negative IC -> underweight US-rate-sensitive assets.")
summary = run_validation(factor, close, factor_id="miner3_20261127_bond_beta_diff_60",
                         regime_notes=notes, return_summary=True)
if summary:
    summary["direction"] = -1
    summary["artifact"] = make_artifact(factor)
    print("SUMMARY_JSON", __import__("json").dumps(summary)[:200])
