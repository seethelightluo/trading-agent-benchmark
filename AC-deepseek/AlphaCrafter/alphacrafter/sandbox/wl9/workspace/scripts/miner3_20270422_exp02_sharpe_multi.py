"""
miner3_20270422_exp02_sharpe_multi_lookback.py
Explore Sharpe ratio with multiple lookback windows (21, 42, 63, 84, 126)
to find optimal risk-adjusted momentum specification.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(".").resolve()))
from scripts.miner3_20260730_harness import (
    load_closes, evaluate, decay_profile
)

LOOKBACKS = [21, 42, 63, 84, 126]
HORIZON = 10

def compute_sharpe_ratio(closes, lookback):
    """Compute rolling Sharpe ratio for each asset."""
    vals = {}
    for a, s in closes.items():
        ret = s.pct_change()
        min_p = max(lookback // 2, 30) if lookback > 60 else max(lookback // 2, 10)
        mean_ret = ret.rolling(lookback, min_periods=min_p).mean()
        std_ret = ret.rolling(lookback, min_periods=min_p).std()
        sharpe = mean_ret / std_ret.replace(0, np.nan)
        vals[a] = sharpe.shift(1)
    return vals

def main():
    print("=== Sharpe Ratio Multi-Lookback Exploration ===\n")
    print(f"Admission horizon: {HORIZON}d forward\n")
    
    closes = load_closes()
    print(f"Loaded {len(closes)} assets\n")
    
    results = []
    for lb in LOOKBACKS:
        vals = compute_sharpe_ratio(closes, lb)
        label = f"sharpe_{lb}d"
        res = evaluate(closes, vals, label, horizon=HORIZON, verbose=False)
        
        print(f"--- {label} ---")
        print(f"  IC(h={HORIZON}): {res['ic']:.4f}  ICIR: {res['icir']:.4f}  Hit: {res['hit']:.3f}")
        print(f"  Coverage: {res['coverage_asset_days']:.3f}  Turnover: {res['turnover_10d_rank']:.3f}")
        
        passes = abs(res['ic']) >= 0.0070 and abs(res['icir']) >= 0.0840
        print(f"  GATE: {'PASS' if passes else 'FAIL'}")
        print(f"  Max_lib_corr: {res['max_abs_library_correlation']:.4f}")
        print()
        
        results.append({
            "label": label,
            "lookback": lb,
            "ic": res["ic"],
            "icir": res["icir"],
            "hit": res["hit"],
            "n_dates": res["n_ic_dates"],
            "coverage": res["coverage_asset_days"],
            "turnover": res["turnover_10d_rank"],
            "max_lib_corr": res["max_abs_library_correlation"],
            "passed": passes
        })
    
    print("\n=== Summary ===")
    print(f"{'Label':<15s} {'IC':>8s} {'ICIR':>8s} {'Hit':>6s} {'Cov':>6s} {'Turn':>6s} {'LibCorr':>8s} {'Gate':>5s}")
    print("-"*70)
    for r in results:
        print(f"{r['label']:<15s} {r['ic']:>8.4f} {r['icir']:>8.4f} {r['hit']:>6.3f} {r['coverage']:>6.3f} {r['turnover']:>6.3f} {r['max_lib_corr']:>8.4f} {'PASS' if r['passed'] else 'FAIL':>5s}")
    
    # Also check what the factor correlates most with
    print("\n=== Best Candidate Detailed Analysis ===")
    best = max(results, key=lambda r: abs(r['icir']) if r['passed'] else -999)
    if best.get('passed'):
        print(f"Best passing: {best['label']} (IC={best['ic']:.4f}, ICIR={best['icir']:.4f})")
    
    print("\nDone.")

if __name__ == "__main__":
    main()