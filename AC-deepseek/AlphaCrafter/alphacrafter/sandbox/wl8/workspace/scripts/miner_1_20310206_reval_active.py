"""miner_1 2031-02-06: re-validate active effective factors (flip_mom_20x10, usdcny_beta_60)
through current visible_through date (2031-02-05). Continuous re-validation cycle.
Also computes library correlation and recent-window drift.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from miner_3_20261203_common import (WATCH, load_prices, load_macro, load_visible_through,
                                     cross_sectional_ic, ic_stats, regime_split,
                                     spearman_panel_rho)

ASOF = load_visible_through()
H = 10
px = load_prices(ASOF)
INDEX = px.index
print(f"ASOF={ASOF} rows={len(INDEX)} px_last={px.index[-1].date()}")

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

fwd = pd.DataFrame({s: forward(px[s], H) for s in WATCH}).sort_index()

def report(name, f):
    icd = cross_sectional_ic(f, fwd)
    st = ic_stats(icd)
    ic252 = ic_stats(icd[icd.index >= icd.index[-1]-pd.Timedelta(days=365)]) if len(icd) else None
    ic180 = ic_stats(icd[icd.index >= icd.index[-1]-pd.Timedelta(days=180)]) if len(icd) else None
    ic60 = ic_stats(icd.tail(60))
    cov = (f.notna() & fwd.notna()).mean().mean()
    print(f"\n==={name}===")
    print(f"FULL: IC={st['ic']:.4f} ICIR={st['icir']:.4f} hit={st['hit']:.3f} n={st['n_dates']} avg_n={st.get('avg_n',np.nan):.1f} cov={cov:.3f}")
    if ic252 is not None: print(f"365d: IC={ic252['ic']:.4f} ICIR={ic252['icir']:.4f} n={ic252['n_dates']}")
    if ic180 is not None: print(f"180d: IC={ic180['ic']:.4f} ICIR={ic180['icir']:.4f} n={ic180['n_dates']}")
    print(f"60d: IC={ic60['ic']:.4f} ICIR={ic60['icir']:.4f} n={ic60['n_dates']}")
    for lab, seg in regime_split(icd).items():
        print(f"  {lab}: IC={seg[0]:.4f} ICIR={seg[1]:.4f} n={seg[2]}")
    for hh in [1,5,10,20]:
        fh = pd.DataFrame({s: forward(px[s], hh) for s in WATCH}).sort_index()
        icd_h = cross_sectional_ic(f, fh)
        print(f"  decay h={hh}: IC={icd_h['ic'].mean():.4f}" if len(icd_h) else f"  decay h={hh}: NA")
    return f, icd

f_flip_o, icd_flip = report("flip_mom_20x10", pd.DataFrame({s: flip_mom(px[s]) for s in WATCH}).sort_index().replace([np.inf,-np.inf], np.nan))

reg = load_macro(ASOF)['USDCNY']
f_beta_o = None
if len(reg) > 0:
    f_beta = pd.DataFrame({s: beta_to(px[s], reg, 60) for s in WATCH}).sort_index().replace([np.inf,-np.inf], np.nan)
    f_beta_o, icb = report("usdcny_beta_60", f_beta)
    print("corr flip_mom vs usdcny_beta (datewise avg rho):", spearman_panel_rho(f_flip_o, f_beta_o))

print("\nGATE: |IC|>=0.0070, |ICIR|>=0.0840")
print("DEPRECATE rule: re-validation FAIL iff recent(60d/180d) IC sign flips or |ICIR| turns clearly negative at full sample")