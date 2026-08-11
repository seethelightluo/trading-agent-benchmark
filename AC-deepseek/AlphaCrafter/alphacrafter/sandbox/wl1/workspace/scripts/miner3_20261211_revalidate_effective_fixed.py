"""miner_3 re-validation of currently EFFECTIVE factors (2026-12-11 cycle) - fixed.

Fixes vix_beta rolling covariance bug: DataFrame.rolling.cov(Series) broadcasts
into a cross-product frame; compute per-asset beta with apply instead.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner3_20261008_lib import load_close_panel, run_validation

close = load_close_panel()
print(f"panel dates={close.shape[0]} assets={close.shape[1]} "
      f"range={close.index.min().date()}..{close.index.max().date()}")

# --- VIX aligned to panel (observation-only macro signal) ---
vix = pd.read_csv("../persistent/index_data/VIX.csv", parse_dates=["date"])
vix = vix.set_index("date")["close"]
cutoff = close.index.max()
vix = vix[vix.index <= cutoff]
vix = vix[~vix.index.duplicated(keep="last")]
vix = vix.reindex(close.index).ffill()
print(f"VIX aligned rows={vix.notna().sum()} range={vix.index.min().date()}..{vix.index.max().date()}")

lr = close.pct_change()

# --- Factor 1: mom_120d_skip5 ---
f1 = close.shift(5) / close.shift(125) - 1.0
run_validation(f1, close, factor_id="mom_120d_skip5 (reval)",
               regime_notes="Re-validation through 2026-12-10; 15-asset universe.")

# --- Factor 2: vol_of_vol20x60 ---
vol20 = lr.rolling(20).std()
f2 = vol20.rolling(60).std()
run_validation(f2, close, factor_id="vol_of_vol20x60 (reval)",
               regime_notes="Re-validation through 2026-12-10; 15-asset universe.")

# --- Factor 3: vix_beta_cond_60x20 (per-asset beta) ---
vix_ret = vix.pct_change()
beta60 = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
for c in close.columns:
    beta60[c] = (close[c].pct_change().rolling(60, min_periods=30)
                 .cov(vix_ret) / vix_ret.rolling(60, min_periods=30).var())
vix_move20 = vix / vix.shift(20) - 1.0
f3 = -beta60.mul(vix_move20, axis=0)
print(f"vix_beta cond valid cells={f3.notna().sum().sum()} "
      f"({f3.notna().mean().mean():.3f} of asset-days)")
run_validation(f3, close, factor_id="vix_beta_cond_60x20 (reval)",
               regime_notes="Re-validation through 2026-12-10; 15-asset universe.")
