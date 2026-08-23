"""miner_3 2031-01-23: re-validate active effective factors (flip_mom_20x10, usdcny_beta_60)
through current visible_through date (2031-01-22). Continuous re-validation cycle.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, json
from miner_3_20261203_common import (WATCH, load_prices, load_macro, load_visible_through,
                                     cross_sectional_ic, ic_stats, regime_split)

ASOF = load_visible_through()
H = 10
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
def flip_mom(p, kw=20, ks=10):
    return (retk(p, kw) * np.sign(retk(p, ks))).reindex(INDEX)
def beta_to(p, reg, window=60):
    a = retk(p, 1); b = retk(reg, 1)
    cov = a.rolling(window).cov(b)
    var = b.rolling(window).var()
    return (cov/var).reindex(INDEX)

add = reg1 = None  # unused placeholders

# ---- flip_mom_20x10 ----
f_flip = pd.DataFrame({s: flip_mom(px[s]) for s in WATCH}).sort_index().replace([np.inf,-np.inf], np.nan)
fwd = pd.DataFrame({s: forward(px[s], H) for s in WATCH}).sort_index()
icd = cross_sectional_ic(f_flip, fwd)
st = ic_stats(icd)
ic252 = ic_stats(icd[icd.index >= icd.index[-1]-pd.Timedelta(days=365)] if len(icd) else icd)
ic60 = ic_stats(icd.tail(60))
cov = (f_flip.notna() & fwd.notna()).mean().mean()
print("===FLIP_MOM_20x10===")
print(f"FULL: IC={st['ic']:.4f} ICIR={st['icir']:.4f} hit={st['hit']:.3f} n={st['n_dates']} avg_n={st.get('avg_n',np.nan):.1f} cov={cov:.3f}")
if len(icd)>7: print(f"365d: IC={ic252['ic']:.4f} ICIR={ic252['icir']:.4f} n={ic252['n_dates']}")
print(f"last60: IC={ic60['ic']:.4f} ICIR={ic60['icir']:.4f} n={ic60['n_dates']}")
for lab, seg in regime_split(icd).items():
    print(f"  {lab}: IC={seg[0]:.4f} ICIR={seg[1]:.4f} n={seg[2]}")
for hh in [1,5,10,20]:
    fh = pd.DataFrame({s: forward(px[s], hh) for s in WATCH}).sort_index()
    icd_h = cross_sectional_ic(f_flip, fh)
    print(f"  decay h={hh}: IC={icd_h['ic'].mean():.4f}" if len(icd_h) else f"  decay h={hh}: NA")

# ---- usdcny_beta_60 ----
reg = load_macro(ASOF)['USDCNY']
if len(reg) > 0:
    f_beta = pd.DataFrame({s: beta_to(px[s], reg, 60) for s in WATCH}).sort_index().replace([np.inf,-np.inf], np.nan)
    icb = cross_sectional_ic(f_beta, fwd)
    sb = ic_stats(icb)
    covb = (f_beta.notna() & fwd.notna()).mean().mean()
    print("===USDCYN_BETA_60===")
    print(f"FULL: IC={sb['ic']:.4f} ICIR={sb['icir']:.4f} hit={sb['hit']:.3f} n={sb['n_dates']} avg_n={sb.get('avg_n',np.nan):.1f} cov={covb:.3f}")
    cb60 = ic_stats(icb.tail(60))
    print(f"last60: IC={cb60['ic']:.4f} ICIR={cb60['icir']:.4f} n={cb60['n_dates']}")
    for lab, seg in regime_split(icb).items():
        print(f"  {lab}: IC={seg[0]:.4f} ICIR={seg[1]:.4f} n={seg[2]}")
    # library overlap
    comm = icd.index.intersection(icb.index)
    rho = pd.concat([icd['ic'], icb['ic']],axis=1).dropna().iloc[:,0].corr(pd.concat([icd['ic'], icb['ic']],axis=1).dropna().iloc[:,1])
    print(f"  co-IC(full flip vs usdcny): {rho:.4f} common={len(comm)}")

print("\nGATE admission: |IC|>=0.0070, |ICIR|>=0.0840")