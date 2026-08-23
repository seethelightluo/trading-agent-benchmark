"""miner_3 2031-02-06: validate mom_diff_20_60 (fast-slow momentum differential).
Compute admission metrics: IC, ICIR, coverage, turnover, decay, and library correlation
vs the 3 ensemble factors + flip_mom. Persist if gate passes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, json
from miner_3_20261203_common import (WATCH, load_prices, load_macro, load_visible_through,
                                     cross_sectional_ic, ic_stats, regime_split, spearman_panel_rho)

ASOF = load_visible_through()
px = load_prices(ASOF)
INDEX = px.index
print(f"ASOF={ASOF} rows={len(INDEX)}")

def vseries(s): return s.dropna()
def retk(s, k):
    v = vseries(s)
    return (v / v.shift(k) - 1.0).reindex(INDEX)
def forward(s, h):
    v = vseries(s)
    return (v.shift(-h)/v - 1.0).reindex(INDEX)

H = 10
fwd = pd.DataFrame({s: forward(px[s], H) for s in WATCH}).sort_index()

def build(df): return df.sort_index().replace([np.inf,-np.inf],np.nan)

f_momdiff = build(pd.DataFrame({s: retk(px[s],20)-retk(px[s],60) for s in WATCH}))
icd = cross_sectional_ic(f_momdiff, fwd)
st = ic_stats(icd)
cov = (f_momdiff.notna() & fwd.notna()).mean().mean()
print(f"===mom_diff_20_60===")
print(f"FULL: IC={st['ic']:+.4f} ICIR={st['icir']:+.4f} hit={st['hit']:.3f} n={st['n_dates']} avg_n={st.get('avg_n',np.nan):.1f} cov={cov:.3f}")

# turnover: mean abs change in cross-sectional rank of factor values
rank = f_momdiff.rank(axis=1)
to = rank.diff().abs().mean().mean()
print(f"turnover_rank: {to:.4f}")

# decay
for hh in [1,3,5,10,20]:
    fh = pd.DataFrame({s: forward(px[s], hh) for s in WATCH}).sort_index()
    icd_h = cross_sectional_ic(f_momdiff, fh)
    print(f"  decay h={hh}: IC={icd_h['ic'].mean():.4f} n={len(icd_h)}" if len(icd_h) else f"  decay h={hh}: NA")

# library correlation vs ensemble factors
library_factors = {}
asm = None
for name, path in [('VIX','../persistent/index_data/VIX.csv'), ('US10Y','../persistent/index_data/../stock_data/US10Y.csv')]:
    pass

# build library factor panels
# mom_10d_skip5-like is not directly in library; use the 3 ensemble factors reconstructed
def beta_cond(p, reg, window=60, cond=20):
    a = retk(p,1); b = retk(reg,1)
    mb = (retk(reg, cond)).reindex(INDEX)
    cov = a.rolling(window).cov(b); var = b.rolling(window).var()
    beta = (cov/var)
    def smreg(x): return x.rolling(window).mean()
    return build(beta * np.sign(mb))

# Load macro for beta factors
mac = load_macro(ASOF)

# mom_10d_skip5: 10d momentum, skip 1 day gap (approximate)
f_mom10 = build(pd.DataFrame({s: retk(px[s],10) for s in WATCH}))
f_vix = beta_cond(px['SPX'], mac['VIX'])  # placeholder signature
f_vix = build(pd.DataFrame({s: beta_cond(px[s], mac['VIX']) for s in WATCH}))
f_yield = build(pd.DataFrame({s: beta_cond(px[s], px['US10Y']) for s in WATCH}))
f_flip = build(pd.DataFrame({s: retk(px[s],20)*np.sign(retk(px[s],10)) for s in WATCH}))

lib = {'mom_10d_skip5': f_mom10, 'vix_beta_cond_60x20': f_vix,
       'yield_beta_cond_60x20': f_yield, 'flip_mom_20x10': f_flip}

maxrho = 0.0; details = {}
for lname, lf in lib.items():
    r = spearman_panel_rho(f_momdiff, lf)
    details[lname] = r
    if np.isfinite(r): maxrho = max(maxrho, abs(r))
print(f"library max_abs_corr: {maxrho:.4f}")
for k,v in details.items(): print(f"  rho vs {k}: {v:.4f}")

admitted = (abs(st['ic'])>=0.0070 and abs(st['icir'])>=0.0840)
print(f"\nGATE: {abs(st['ic']):.4f}>=0.0070? {abs(st['ic'])>=0.0070} | {abs(st['icir']):.4f}>=0.0840? {abs(st['icir'])>=0.0840} -> ADMITTED={admitted}")