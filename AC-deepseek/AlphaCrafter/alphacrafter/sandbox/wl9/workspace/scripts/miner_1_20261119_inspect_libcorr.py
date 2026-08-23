"""Inspect library correlation detail for clv_trend_20 and sortino_20."""
from __future__ import annotations
import sys
sys.path.insert(0, "scripts")
import numpy as np
from miner3_20260730_harness import (
    FACTOR_DIR, evaluate, load_closes, library_correlation, to_frame, forward_returns, rank_ic,
)

def compute_clv_trend(closes):
    vals = {}
    for a, s in closes.items():
        hi = s.rolling(20).max()
        lo = s.rolling(20).min()
        rng = (hi - lo).replace(0, np.nan)
        clv = (s - lo) / rng
        m20 = s / s.shift(20) - 1.0
        vals[a] = (clv * np.sign(m20)).shift(1)
    return vals

def compute_sortino(closes):
    vals = {}
    for a, s in closes.items():
        r = s.pct_change()
        mu = r.rolling(20).mean()
        neg = r.clip(upper=0)
        dsd = neg.rolling(20).std().replace(0, np.nan)
        vals[a] = (mu / dsd).shift(1)
    return vals

closes = load_closes()
for name, fn in [("clv_trend_20", compute_clv_trend), ("sortino_20", compute_sortino)]:
    vals = fn(closes)
    frame = to_frame(closes, vals)
    maxr, detail = library_correlation(frame)
    print(f"=== {name} max_abs_lib_corr={maxr:.4f}")
    for k, v in sorted(detail.items(), key=lambda x: -abs(x[1])):
        print(f"   {k}: {v}")