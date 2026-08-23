"""
miner1_20270506_exp01_rank_momentum.py
Explore cross-sectional rank momentum: rank assets by their recent total return,
then use the rank (normalized to [0,1]) as the factor signal. This normalizes
across diverse asset classes (equities 12%/yr vol, crypto 60%/yr, bonds 5%/yr)
so signals are comparable.

Factor: rank_mom_20d = cross-sectional rank of 20d total return
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys, json, base64, io, zlib

sys.path.insert(0, str(Path(".").resolve()))
from scripts.miner3_20260730_harness import (
    load_closes, load_macro, to_frame, forward_returns,
    rank_ic, turnover_rank10, decay_profile, library_correlation, evaluate
)

HORIZONS = [1, 2, 3, 5, 10, 20]
WINDOWS = [10, 20, 40, 60]

def compute_cs_rank_momentum(closes, lookback):
    """Compute cross-sectional rank of total return."""
    vals = {}
    returns = {}
    for a, s in closes.items():
        ret = s.pct_change(lookback)
        returns[a] = ret.shift(1)
    
    # Build frame and compute cross-sectional ranks
    ret_df = pd.DataFrame(returns)
    rank_df = ret_df.rank(axis=1, pct=True)  # normalized ranks 0..1
    # Set NaN where returns are NaN
    rank_df[ret_df.isna()] = np.nan
    
    for a in closes:
        vals[a] = rank_df[a]
    return vals

def main():
    print("=== Cross-Sectional Rank Momentum Exploration ===\n")
    closes = load_closes()
    print(f"Loaded {len(closes)} assets\n")
    
    results_summary = {}
    
    for w in WINDOWS:
        label = f"cs_rank_mom_{w}d"
        print(f"\n{'='*60}")
        print(f"  WINDOW = {w}d")
        print(f"{'='*60}")
        
        factor_vals = compute_cs_rank_momentum(closes, w)
        
        for h in HORIZONS:
            result = evaluate(closes, factor_vals, label, horizon=h, verbose=False)
            ic_val = result["ic"]
            icir_val = result["icir"]
            n_dates = result.get("n_ic_dates", 0)
            coverage = result.get("coverage_asset_days", 0)
            
            print(f"  h={h:>2d}: IC={ic_val:+.6f}  ICIR={icir_val:+.6f}  n_dates={n_dates}  coverage={coverage:.3f}")
            
            if h not in results_summary:
                results_summary[h] = {}
            results_summary[h][label] = {"ic": ic_val, "icir": icir_val, "n": n_dates, "cov": coverage}
    
    # Best results
    print("\n\n=== BEST CONFIGURATIONS ===")
    for h in HORIZONS:
        best_label = max(results_summary[h], key=lambda k: abs(results_summary[h][k]["ic"]))
        r = results_summary[h][best_label]
        print(f"  h={h:>2d}: {best_label}  IC={r['ic']:+.6f}  ICIR={r['icir']:+.6f}")
    
    # Check admission at h=10
    print("\n\n=== ADMISSION GATE h=10 ===")
    for label in [f"cs_rank_mom_{w}d" for w in WINDOWS]:
        r = results_summary[10].get(label, {})
        ic, icir = r.get("ic", 0), r.get("icir", 0)
        gate_ic = abs(ic) >= 0.007
        gate_icir = abs(icir) >= 0.084
        print(f"  {label:25s}: IC={ic:+.6f} ICIR={icir:+.6f}  |IC|>={gate_ic}  |ICIR|>={gate_icir}  PASS={gate_ic and gate_icir}")


if __name__ == "__main__":
    main()