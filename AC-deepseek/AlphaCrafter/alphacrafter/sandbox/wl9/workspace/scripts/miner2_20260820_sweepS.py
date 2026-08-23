"""miner_2 (2026-08-20): explore variations orthogonal to existing library.

Focus: candidates combining momentum/vol with regime separation, targeting
low correlation with bb_width_20d/kaufman_eff.

1. mom_z_20: rolling 20d return z-scored vs its own 120d distribution.
2. vol_ratio_20_120: short vs long vol ratio (already tried 10/60, try 20/120).
3. drawup_speed_10: close recovery relative to 10d min (bounce strength after dip).
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import evaluate, load_closes


def mom_z(close, w=20, base=120):
    m = close / close.shift(w) - 1.0
    r = close.pct_change()
    sd = r.rolling(base, min_periods=base // 2).std() * np.sqrt(w)
    return m / sd.replace(0, np.nan)


def vol_ratio(close, n1=20, n2=120):
    r = close.pct_change()
    s1 = r.rolling(n1, min_periods=n1 // 2).std()
    s2 = r.rolling(n2, min_periods=n2 // 2).std()
    return s1 / s2


def drawup(close, n=10):
    return close / close.shift(n) - 1.0


def main():
    closes = load_closes()
    cands = {
        "mom_z_20_120": {a: mom_z(s, 20, 120) for a, s in closes.items()},
        "vol_ratio_20_120": {a: vol_ratio(s, 20, 120) for a, s in closes.items()},
        "drawup_10d": {a: drawup(s, 10) for a, s in closes.items()},
    }
    for name, vals in cands.items():
        try:
            res = evaluate(closes, vals, name, horizon=10)
            print(f"RESULT {name}: IC={res['ic']:.4f} ICIR={res['icir']:.4f} "
                  f"max_corr={res['max_abs_library_correlation']:.4f} passed={res['passed']}\n")
        except Exception as e:
            print(f"ERROR {name}: {e}")


if __name__ == "__main__":
    main()