"""Debug: why beta factors / z-composites are empty in batch 3."""
import sys
sys.path.insert(0, "scripts")
import pandas as pd
import numpy as np
from factor_harness import get_panels, evaluate, WATCH

closes, rets, ohlc, macro = get_panels()
mclose = macro
vix = mclose["VIX"]

def roll_beta(y, x, n=60):
    dy = y.diff()
    dx = x.diff()
    cov = dy.rolling(n).cov(dx)
    var = dx.rolling(n).var()
    return cov / var

b = roll_beta(closes, vix)
print("beta shape:", b.shape)
print("beta NaN frac:", b.isna().mean().mean())
print("beta head cols:", list(b.columns)[:3])
print(b.tail(3))

# check z-composite
def zscore(px):
    return (px - px.mean(axis=1)) / px.std(axis=1)
mom20 = closes.pct_change(20)
z20 = zscore(mom20)
print("\nz20 shape:", z20.shape, "NaN frac:", z20.isna().mean().mean())
print(z20.tail(2))
