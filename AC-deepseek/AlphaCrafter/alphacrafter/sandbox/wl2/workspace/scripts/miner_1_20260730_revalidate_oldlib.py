"""miner_1 2026-07-30 cycle 17: re-validation of previously quarantined library
factors (mom_10d_skip5, mom_120d_skip5, vol_of_vol20x60, vix_beta_cond_60x20)
in the EXACT gate namespace (close + pd + np on the UNION panel).

Purpose: determine which old factors can be re-admitted (recoverable artifact +
passing IC/ICIR) vs which must remain deprecated (no recoverable artifact).
Also probes coverage of rolling-based expressions to document why they collapse.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner3_lib import (build_panel, forward_returns, spearman_ic,
                        mean_rank_turnover, ADMISSION_HORIZON, MIN_ASSETS)

prices = build_panel()
panel = pd.DataFrame(prices)
env = {"pd": pd, "np": np, "close": panel}

EXPRS = {
    # old library, shift-based (potentially recoverable)
    "mom_10d_skip5":    "close.shift(5) / close.shift(15) - 1.0",
    "mom_120d_skip5":   "close.shift(5) / close.shift(125) - 1.0",
    # old library, rolling-based (previously quarantined)
    "vol_of_vol20x60":  "close.pct_change().rolling(20).std().rolling(60).std()",
    # rolling-based probes (document coverage collapse)
    "vol_20d":          "close.pct_change().rolling(20).std()",
    "zscore_60d":       "(close - close.rolling(60).mean()) / close.rolling(60).std()",
}
fwd10 = forward_returns(prices, ADMISSION_HORIZON)
print("=== old-library re-validation in gate namespace ===")
for fid, exp in EXPRS.items():
    try:
        sig = eval(exp, {"__builtins__": {}}, env)
        ok = isinstance(sig, pd.DataFrame) and sig.shape == panel.shape
        n_valid = int(sig.notna().sum().sum()) if ok else 0
        cov = n_valid / sig.size if ok else 0.0
        n_ge8 = sum(1 for d in sig.index if sig.loc[d].notna().sum() >= MIN_ASSETS) if ok else 0
        print(f"  {fid:18s} eval={'OK' if ok else 'BAD'} valid_cells={n_valid} "
              f"cov_asset_days={cov:.3f} dates_ge8={n_ge8}/{len(panel)}")
        if ok and n_valid > 100:
            ic_ser = spearman_ic(sig, fwd10)
            ic = float(ic_ser.mean())
            icir = float(ic_ser.mean() / ic_ser.std()) if ic_ser.std() > 0 else 0.0
            print(f"             n_ic={len(ic_ser)} ic={ic:+.4f} icir={icir:+.4f} "
                  f"gate={'PASS' if abs(ic)>=0.007 and abs(icir)>=0.084 else 'FAIL'}")
    except Exception as e:
        print(f"  {fid:18s} eval=FAIL {type(e).__name__}: {str(e)[:70]}")

# vix_beta is not expressible at all in {close,pd,np} namespace (needs VIX macro)
print("\n  vix_beta_cond_60x20: NOT expressible in gate namespace (requires VIX macro series) -> no recoverable artifact")
