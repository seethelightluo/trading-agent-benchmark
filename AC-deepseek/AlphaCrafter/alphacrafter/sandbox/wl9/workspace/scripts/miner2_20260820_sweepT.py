"""miner_2 (2026-08-20): explore orthogonal momentum/regime variants.

Target: retain the strong mom_align signal but reduce correlation with
bb_width_20d / kaufman_eff below 0.5.

1. mom_align_rank_5_20_60: use rank-sum of momentum buckets (slower/cleaner).
2. mom_align_5_20: two-window alignment, momentum-magnitude-weighted.
3. hilo_pos_60: close relative to 60d (high+low)/2 -> trend position indicator.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import evaluate, load_closes


def mom_align_rank(close):
    ms = []
    for wk in (5, 20, 60):
        m = close / close.shift(wk) - 1.0
        ms.append(m.rolling(20, min_periods=5).rank(pct=True) - 0.5)
    rsum = ms[0] + ms[1] * 2.0 + ms[2] * 3.0
    return rsum


def mom_align_two(close, w1=5, w2=20):
    m1 = close / close.shift(w1) - 1.0
    m2 = close / close.shift(w2) - 1.0
    al = np.sign(m1) * np.sign(m2) * (m1 + m2) / 2.0
    return al


def hilo_pos(close, n=60):
    hi = close.rolling(n, min_periods=n // 2).max()
    lo = close.rolling(n, min_periods=n // 2).min()
    return (close - (hi + lo) / 2.0) / (hi - lo).replace(0, np.nan)


def main():
    closes = load_closes()
    cands = {
        "mom_align_rank_5_20_60": {a: mom_align_rank(s) for a, s in closes.items()},
        "mom_align_two_5_20": {a: mom_align_two(s, 5, 20) for a, s in closes.items()},
        "hilo_pos_60": {a: hilo_pos(s, 60) for a, s in closes.items()},
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