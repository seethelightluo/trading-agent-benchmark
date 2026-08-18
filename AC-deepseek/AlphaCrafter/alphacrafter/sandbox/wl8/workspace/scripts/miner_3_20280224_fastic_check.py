"""Vectorized date-wise spearman IC implementation + equivalence check vs loop version."""
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
    """Vectorized date-wise Spearman IC (Pearson on cross-sectional ranks)."""
    f = fdf.rank(axis=1)
    r = rdf.rank(axis=1)
    mask = fdf.notna() & rdf.notna() & np.isfinite(fdf) & np.isfinite(rdf)
    n = mask.sum(axis=1)
    f = f.where(mask)
    r = r.where(mask)
    fm = f.sub(f.mean(axis=1), axis=0)
    rm = r.sub(r.mean(axis=1), axis=0)
    num = (fm * rm).sum(axis=1)
    den = np.sqrt((fm ** 2).sum(axis=1) * (rm ** 2).sum(axis=1))
    ic = num / den.replace(0, np.nan)
    ic = ic.where(n >= min_assets)
    return ic.dropna()

# test factor
fz = np.random.RandomState(0).randn(len(INDEX), len(WATCH))
fdf = pd.DataFrame(fz, index=INDEX, columns=WATCH)
fdf.iloc[::7, 0] = np.nan
fwd = fwd_panel(10)

loop_ic = cross_sectional_ic(fdf, fwd)
fast_ic_s = fast_ic(fdf, fwd)
print('loop dates:', len(loop_ic), 'fast dates:', len(fast_ic_s))
if len(loop_ic) == len(fast_ic_s):
    d = (loop_ic['ic'] - fast_ic_s).abs().max()
    print('max abs diff:', d)
    print('mean loop:', loop_ic['ic'].mean(), 'mean fast:', fast_ic_s.mean())
else:
    # align
    a = loop_ic['ic']
    b = fast_ic_s
    print('index mismatch; aligned:', len(a.index.intersection(b.index)))