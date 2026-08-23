"""
miner3_20270422_exp04_price_acceleration.py
Explore price acceleration factor: the rate of change of momentum.
Factor definitions:
1. accel_5_20 = r(5) - r(20): short-rate acceleration
2. accel_10_30 = r(10) - r(30): medium-rate acceleration
3. accel_10_50 = r(10) - r(50): longer comparison
4. accel_ratio_5_20 = r(5)/r(20) - 1: ratio version
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(".").resolve()))
from scripts.miner3_20260730_harness import load_closes, evaluate

HORIZON = 10

def main():
    print("=== Price Acceleration Factor Exploration ===\n")
    closes = load_closes()
    print(f"Loaded {len(closes)} assets\n")
    candidates = {}
    # 1. accel_5_20: 5d ret minus 20d ret
    vals1 = {}
    for a, s in closes.items():
        r5 = s.pct_change(5)
        r20 = s.pct_change(20)
        accel = r5 - r20
        vals1[a] = accel.shift(1)
    candidates["accel_5_20"] = vals1
    # 2. accel_10_30: 10d ret minus 30d ret
    vals2 = {}
    for a, s in closes.items():
        r10 = s.pct_change(10)
        r30 = s.pct_change(30)
        accel = r10 - r30
        vals2[a] = accel.shift(1)
    candidates["accel_10_30"] = vals2
    # 3. accel_5_40: 5d ret minus 40d ret
    vals3 = {}
    for a, s in closes.items():
        r5 = s.pct_change(5)
        r40 = s.pct_change(40)
        accel = r5 - r40
        vals3[a] = accel.shift(1)
    candidates["accel_5_40"] = vals3
    # 4. accel_10_50: 10d ret minus 50d ret
    vals4 = {}
    for a, s in closes.items():
        r10 = s.pct_change(10)
        r50 = s.pct_change(50)
        accel = r10 - r50
        vals4[a] = accel.shift(1)
    candidates["accel_10_50"] = vals4
    # Run evaluations
    results = []
    for label, vals in candidates.items():
        res = evaluate(closes, vals, label, horizon=HORIZON, verbose=False)
        passes = abs(res['ic']) >= 0.0070 and abs(res['icir']) >= 0.0840
        results.append({
            "label": label, "ic": res["ic"], "icir": res["icir"],
            "hit": res["hit"], "n_dates": res["n_ic_dates"],
            "coverage": res["coverage_asset_days"],
            "turnover": res["turnover_10d_rank"],
            "max_lib_corr": res["max_abs_library_correlation"],
            "passed": passes
        })
        print(f"{label:20s}: IC={res['ic']:.4f} ICIR={res['icir']:.4f} "
              f"Hit={res['hit']:.3f} Cov={res['coverage_asset_days']:.3f} "
              f"Turn={res['turnover_10d_rank']:.3f} "
              f"LibCorr={res['max_abs_library_correlation']:.4f} "
              f"{'PASS' if passes else 'FAIL'}")
    print("\n=== Summary ===")
    print(f"{'Label':20s} {'IC':>8s} {'ICIR':>8s} {'Hit':>6s} {'Cov':>6s} {'Turn':>6s} {'LCorr':>8s} {'Gate':>5s}")
    print("-"*75)
    for r in results:
        print(f"{r['label']:20s} {r['ic']:>8.4f} {r['icir']:>8.4f} {r['hit']:>6.3f} "
              f"{r['coverage']:>6.3f} {r['turnover']:>6.3f} {r['max_lib_corr']:>8.4f} "
              f"{'PASS' if r['passed'] else 'FAIL':>5s}")
    print("\nDone.")
if __name__ == "__main__":
    main()