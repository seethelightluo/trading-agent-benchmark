"""miner_1 (2026-10-22): price-structure / trend-quality family exploration.

Ideas (all price/return based, no volume since volume is constant, no fundamentals):
  C1 clv_mom_20    : close-location-value (where close sits in 20d high-low range)
                     combined with 20d momentum -> trend-quality tilt
  C2 upside_ratio_20: fraction of up days over 20d (return hit-rate / breadth)
  C3 roll_sharpe_20 : rolling 20d Sharpe ratio of daily returns
  C4 mom_accel_10_20: momentum acceleration = 20d momentum - 10d momentum
  C5 range_skew_pos  : running skewness of signed daily range contribution
Uses shared harness (miner3_20260730_harness).
Gate on 15-asset universe: abs(IC)>=0.0070 & abs(ICIR)>=0.0840 @ h=10.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner3_20260730_harness import load_closes, evaluate


def clv(close, high, low, w):
    hi = high.rolling(w).max()
    lo = low.rolling(w).min()
    rng = (hi - lo).replace(0, np.nan)
    return (close - lo) / rng


def main():
    closes = load_closes()
    print(f"assets loaded: {len(closes)}")

    # C1: close-location-value momentum (trend-quality)
    vals = {}
    for a, s in closes.items():
        r = s.pct_change()
        mom20 = s / s.shift(20) - 1.0
        clv20 = clv(s, None, None, 20)
        # fallback: compute clv from close vs rolling high/low of close series
        hi = s.rolling(20).max()
        lo = s.rolling(20).min()
        clv20 = (s - lo) / (hi - lo).replace(0, np.nan)
        vals[a] = (clv20 * np.sign(mom20)).shift(1)
    evaluate(closes, vals, "C1 clv_mom_20 (clv20*sign(mom20))")

    # C2: upside_ratio_20 (breadth: fraction of up days over 20d)
    vals = {}
    for a, s in closes.items():
        r = s.pct_change()
        up = (r > 0).rolling(20).mean()
        vals[a] = up.shift(1)
    evaluate(closes, vals, "C2 upside_ratio_20 (20d up-day fraction)")

    # C3: rolling Sharpe ratio 20d
    vals = {}
    for a, s in closes.items():
        r = s.pct_change()
        mu = r.rolling(20).mean()
        sd = r.rolling(20).std().replace(0, np.nan)
        vals[a] = (mu / sd).shift(1)
    evaluate(closes, vals, "C3 roll_sharpe_20")

    # C4: momentum acceleration = mom20 - mom10
    vals = {}
    for a, s in closes.items():
        m20 = s / s.shift(20) - 1.0
        m10 = s / s.shift(10) - 1.0
        vals[a] = (m20 - m10).shift(1)
    evaluate(closes, vals, "C4 mom_accel_10_20 (m20-m10)")

    # C5: signed range skew (fragility) = skew of daily up/down contribution
    vals = {}
    for a, s in closes.items():
        r = s.pct_change()
        vals[a] = r.rolling(20).skew().shift(1)
    evaluate(closes, vals, "C5 ret_skew_20 (rolling 20d return skewness)")


if __name__ == "__main__":
    main()
