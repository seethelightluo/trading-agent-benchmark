"""miner_3 cycle 2026-12-31: reproduce NaN/coverage + dict-key bug in screen pipeline."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, json
from miner_3_20261203_common import WATCH, MACRO, load_prices, load_macro, load_visible_through

asof = load_visible_through()
print("visible_through =", asof)
px = load_prices(asof)
macro = load_macro(asof)
print("px shape:", px.shape, "date range:", px.index.min(), "->", px.index.max())
print("px NaN fraction: %.4f" % px.isna().mean().mean())
print("NaN per column (tail 15 rows):")
print(px.isna().tail(15).sum())
print("pct_change NaN fraction: %.4f" % px.pct_change().isna().mean().mean())

fwd = px.shift(-10) / px - 1.0
print("fwd NaN fraction: %.4f" % fwd.isna().mean().mean())

# quick rolling std coverage check with strict vs relaxed min_periods
r = px.pct_change()
for mp in [12, 6, 3, 1, 0]:
    cov = r.rolling(20, min_periods=mp).std()
    print(f"rolling std 20 mp={mp}: NaN frac {cov.isna().mean().mean():.4f}, last-row valid assets {cov.iloc[-1].notna().sum()}")

# union vs intersect calendar check: how many dates have >=8 assets with valid close?
valid = px.notna()
ge8 = (valid.sum(axis=1) >= 8).mean()
print("dates with >=8 valid closes: %.3f" % ge8, "of", len(px))

# check dict-key error source: regime_split keys / NaN in rho dict
cn = px['000300.SH']
try:
    rv = px.pct_change().rolling(20).std()
    ratio = rv.div(cn.pct_change().rolling(20).std(), axis=0)
    print("vol ratio tail NaN frac:", ratio.isna().mean().mean())
except Exception as e:
    print("vol ratio error:", e)
print("done")