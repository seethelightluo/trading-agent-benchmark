"""miner_1 exploration: relative volatility spread family.

Idea: cross-sectional differences in *relative* realized volatility expansion
(short-horizon RV vs medium-run RV regime) predict next-10d cross-asset returns.
Also test short-window amplitude ratio and intraday-range skewness variants.
Uses shared harness (miner3_20260730_harness) for IC/ICIR/turnover/decay and
library correlation to avoid persisting near-duplicates.
Admission gate on 15-asset universe: abs(IC)>=0.0070 & abs(ICIR)>=0.0840 @ h=10.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner3_20260730_harness import load_closes, load_macro, evaluate


def rvol(close, w):
    r = close.pct_change()
    return r.rolling(w).std()


def main():
    closes = load_closes()
    print(f"assets loaded: {len(closes)}")
    # ---- Candidate 1: rvol_spread_20_60 ----
    # relative realized vol expansion: z-score of vol20/vol60 ratio
    vals = {}
    for a, s in closes.items():
        r = s.pct_change()
        v20 = r.rolling(20).std()
        v60 = r.rolling(60).std()
        ratio = v20 / v60
        z = (ratio - ratio.rolling(120).mean()) / ratio.rolling(120).std()
        vals[a] = z.shift(1)  # ensure no same-day lookahead
    evaluate(closes, vals, "CAND1 rvol_spread_20_60 (z(vol20/vol60), 120d center)")

    # ---- Candidate 2: amp_ratio_5_20 ----
    # short-term (high-low)/close amplitude vs its 20d mean -> amplitude compression/expansion
    vals = {}
    for a, s in closes.items():
        hl = (s.rolling(5).max() - s.rolling(5).min()) / s
        base = hl.rolling(20).mean()
        vals[a] = (hl / base).shift(1)
    evaluate(closes, vals, "CAND2 amp_ratio_5_20 (5d amplitude / 20d mean)")

    # ---- Candidate 3: sk_hl_20 ----
    # skewness of daily (high-low)/close over 20d: positive = occasional wide-range days (fragility)
    vals = {}
    for a, s in closes.items():
        hl = (s.high - s.low) / s.close if "high" in s and "low" in s and "close" in s else None
        if hl is None:
            vals[a] = np.nan
            continue
        vals[a] = hl.rolling(20).skew().shift(1)
    evaluate(closes, vals, "CAND3 sk_daily_range_20 (skew of high-low/close, 20d)")


if __name__ == "__main__":
    main()