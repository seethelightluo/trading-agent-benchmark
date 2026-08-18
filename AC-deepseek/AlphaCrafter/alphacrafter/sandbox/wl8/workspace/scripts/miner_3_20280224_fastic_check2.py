"""Vectorized date-wise spearman IC v2 (rank AFTER masking) + equivalence vs loop."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from miner_3_20261203_common import WATCH, load_prices, load_macro, cross_sectional_ic

ASOF = '2028-02-23'
px = load_prices(ASOF)
INDEX = px.index

def fwd_panel(h):
    out = {}
    for s in WATCH:
        v = px[s].dropna()
        out[s] = (v.shift(-h) / v - 1.0).reindex(INDEX)
    return pd.DataFrame(out).sort_index()

def fast_ic(fdf, rdf, min_assets=8):
    mask = fdf.notna() & rdf.notna() & np.isfinite(fdf) & np.isfinite(rdf)
    n = mask.sum(axis=1)
    f = fdf.where(mask).rank(axis=1)
    r = rdf.where(mask).rank(axis=1)
    fm = f.sub(f.mean(axis=1), axis=0)
    rm = r.sub(r.mean(axis=1), axis=0)
    num = (fm * rm).sum(axis=1)
    den = np.sqrt((fm ** 2).sum(axis=1) * (rm ** 2).sum(axis=1))
    ic = (num / den.replace(0, np.nan)).where(n >= min_assets)
    return ic.dropna()

# test on real-ish factor
fdf = pd.DataFrame(np.random.RandomState(0).randn(len(INDEX), len(WATCH)), index=INDEX, columns=WATCH)
fdf.iloc[::7, 0] = np.nan
fdf.iloc[::11, 3] = np.nan
fdf.iloc[-5:, :] = np.nan
fwd = fwd_panel(10)

loop_ic = cross_sectional_ic(fdf, fwd)
fast_s = fast_ic(fdf, fwd)
a = loop_ic['ic']; b = fast_s
print('loop dates:', len(a), 'fast dates:', len(b))
idx = a.index.intersection(b.index)
print('aligned:', len(idx))
if len(idx):
    print('max abs diff:', (a.loc[idx] - b.loc[idx]).abs().max())
    print('mean loop:', a.mean(), 'mean fast:', b.mean())
# also test with NaN in forward returns (end of sample)