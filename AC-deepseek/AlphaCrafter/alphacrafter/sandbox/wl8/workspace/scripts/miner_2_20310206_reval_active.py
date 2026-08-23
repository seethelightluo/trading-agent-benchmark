"""miner_2 2031-02-06: re-validate active effective factors (flip_mom_20x10, usdcny_beta_60)
through current visible_through date (2031-02-05). Continuous re-validation / drift check.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from miner_3_20261203_common import WATCH, load_prices, load_macro, load_visible_through, \
     cross_sectional_ic, ic_stats, regime_split

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

fwd10 = pd.DataFrame({s: forward(px[s], H) for s in WATCH}).sort_index()
print(f"ASOF={ASOF} rows={len(INDEX)} horizon={H}")
print("GATE admission: |IC|>=0.0070, |ICIR|>=0.0840")

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
        for lab, seg in regime_split(icd, breaks=(pd.Timestamp('2022-01-01'), pd.Timestamp('2024-01-01'))).items():
            print(f"  {lab}: IC={seg[0]:.4f} ICIR={seg[1]:.4f} n={seg[2]}")
        # recent-year splits
        for lab, m in [('2027', icd.index<pd.Timestamp('2028-01-01')),
                       ('2028', (icd.index>=pd.Timestamp('2028-01-01'))&(icd.index<pd.Timestamp('2029-01-01'))),
                       ('2029', (icd.index>=pd.Timestamp('2029-01-01'))&(icd.index<pd.Timestamp('2030-01-01'))),
                       ('2030', (icd.index>=pd.Timestamp('2030-01-01'))&(icd.index<pd.Timestamp('2031-01-01'))),
                       ('2031YTD', icd.index>=pd.Timestamp('2031-01-01'))]:
            sub = icd[m]
            if len(sub):
                ss = ic_stats(sub)
                print(f"  Year {lab}: IC={ss['ic']:.4f} ICIR={ss['icir']:.4f} n={ss['n_dates']}")
    for hh in [1,5,10,20]:
        fh = pd.DataFrame({s: forward(px[s], hh) for s in WATCH}).sort_index()
        icd_h = cross_sectional_ic(fdf, fh)
        print(f"  decay h={hh}: IC={icd_h['ic'].mean():.4f}" if len(icd_h) else f"  decay h={hh}: NA")
    return icd

f_flip = pd.DataFrame({s: flip_mom(px[s]) for s in WATCH}).sort_index().replace([np.inf,-np.inf], np.nan)
report("FLIP_MOM_20x10 (dir=1)", f_flip)

reg = load_macro(ASOF)['USDCNY']
f_beta = pd.DataFrame({s: beta_to(px[s], reg, 60) for s in WATCH}).sort_index().replace([np.inf,-np.inf], np.nan)
report("USDCYN_BETA_60 (dir=1)", f_beta)

# library correlation between the two
from miner_3_20261203_common import spearman_panel_rho
print(f"\nlib rho (flip_mom, usdcny_beta) = {spearman_panel_rho(f_flip, f_beta):.4f}")