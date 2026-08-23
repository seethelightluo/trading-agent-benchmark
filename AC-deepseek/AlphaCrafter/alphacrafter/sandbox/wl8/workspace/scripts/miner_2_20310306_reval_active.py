"""miner_2 2031-03-06: re-validate active effective factors through 2031-03-05.
Continuous re-validation / drift check on flip_mom_20x10, usdcny_beta_60, mom_diff_20_60.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from miner_3_20261203_common import WATCH, load_prices, load_macro, load_visible_through, \
     cross_sectional_ic, ic_stats, regime_split, spearman_panel_rho

ASOF = load_visible_through()
H = 10
px = load_prices(ASOF)
INDEX = px.index

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
def mom_diff(p, kw=20, ks=60):
    return (retk(p, kw) - retk(p, ks)).reindex(INDEX)

fwd10 = pd.DataFrame({s: forward(px[s], H) for s in WATCH}).sort_index()
print(f"ASOF={ASOF} rows={len(INDEX)} horizon={H}")
print("GATE: |IC|>=0.0070, |ICIR|>=0.0840")

def report(label, fdf):
    icd = cross_sectional_ic(fdf, fwd10)
    st = ic_stats(icd)
    cov = (fdf.notna() & fwd10.notna()).mean().mean()
    print(f"\n=== {label} ===")
    print(f"FULL: IC={st['ic']:.4f} |IC|={abs(st['ic']):.4f} ICIR={st['icir']:.4f} |ICIR|={abs(st['icir']):.4f} hit={st['hit']:.3f} n={st['n_dates']} avg_n={st.get('avg_n',np.nan):.1f} cov={cov:.3f}")
    if len(icd):
        ic365 = ic_stats(icd[icd.index >= icd.index[-1]-pd.Timedelta(days=365)])
        print(f"365d:   IC={ic365['ic']:.4f} ICIR={ic365['icir']:.4f} hit={ic365['hit']:.3f} n={ic365['n_dates']}")
        ic60 = ic_stats(icd.tail(60))
        print(f"last60: IC={ic60['ic']:.4f} ICIR={ic60['icir']:.4f} hit={ic60['hit']:.3f} n={ic60['n_dates']}")
        for lab, seg in regime_split(icd).items():
            print(f"  {lab}: IC={seg[0]:.4f} ICIR={seg[1]:.4f} n={seg[2]}")
    for hh in [1,5,10,20]:
        fh = pd.DataFrame({s: forward(px[s], hh) for s in WATCH}).sort_index()
        icd_h = cross_sectional_ic(fdf, fh)
        if len(icd_h):
            print(f"  decay h={hh}: IC={icd_h['ic'].mean():.4f}")
    return icd

f_flip = pd.DataFrame({s: flip_mom(px[s]) for s in WATCH}).sort_index().replace([np.inf,-np.inf], np.nan)
icd = report("FLIP_MOM_20x10 (dir=1)", f_flip)

reg = load_macro(ASOF)['USDCNY']
f_beta = pd.DataFrame({s: beta_to(px[s], reg, 60) for s in WATCH}).sort_index().replace([np.inf,-np.inf], np.nan)
icd2 = report("USDCYN_BETA_60 (dir=1)", f_beta)

f_md = pd.DataFrame({s: mom_diff(px[s]) for s in WATCH}).sort_index().replace([np.inf,-np.inf], np.nan)
icd3 = report("MOM_DIFF_20_60 (dir=1)", f_md)

print(f"\nlib rho (flip, usdcny) = {spearman_panel_rho(f_flip, f_beta):.4f}")
print(f"lib rho (flip, momdiff) = {spearman_panel_rho(f_flip, f_md):.4f}")
print(f"lib rho (usdcny, momdiff) = {spearman_panel_rho(f_beta, f_md):.4f}")