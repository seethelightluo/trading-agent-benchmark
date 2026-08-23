"""miner_2 (2026-08-20): explore three new factor ideas.

1. dd_120d: drawdown depth = close/rolling_max(close,120) - 1 (regime/recovery).
2. vol_term_10_60: vol term-structure slope = (sd10/sd60), a 'volatility contango'.
3. mom_align_5_20_60: alignment of short/mid/long momentum signs (trend quality).

Only reports metrics; persistence decided manually.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import evaluate, load_closes


def dd_depth(close, n=120):
    return close / close.rolling(n, min_periods=n // 2).max() - 1.0


def vol_term(close, n1=10, n2=60):
    r = close.pct_change()
    s1 = r.rolling(n1, min_periods=n1 // 2).std()
    s2 = r.rolling(n2, min_periods=n2 // 2).std()
    return s1 / s2


def mom_align(close, w=(5, 20, 60)):
    ms = [np.sign(close / close.shift(wk) - 1.0) for wk in w]
    al = np.sign(ms[0] + ms[1] + ms[2])
    return al * (close / close.shift(w[1]) - 1.0)


def main():
    closes = load_closes()
    cands = {
        "dd_120d": {a: dd_depth(s, 120) for a, s in closes.items()},
        "vol_term_10_60": {a: vol_term(s, 10, 60) for a, s in closes.items()},
        "mom_align_5_20_60": {a: mom_align(s) for a, s in closes.items()},
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