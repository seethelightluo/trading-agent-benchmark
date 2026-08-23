"""miner_2 2030-08-08: re-validate current effective library factors (flip_mom_20x10, usdcny_beta_60)
through current visible date 2030-08-07 for drift/timeliness."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, json
from miner_3_20261203_common import WATCH, load_prices, load_macro, load_visible_through, cross_sectional_ic, ic_stats

ASOF = load_visible_through()
H = 10
px = load_prices(ASOF)
INDEX = px.index

def vseries(s): return s.dropna()
def retk(s, k):
    v = vseries(s)
    return (v / v.shift(k) - 1.0).reindex(INDEX)

def flip_mom(p, kw=20, ks=10):
    return (retk(p, kw) * np.sign(retk(p, ks))).reindex(INDEX)

def fwd_h(h):
    return pd.DataFrame({s: (vseries(px[s]).shift(-h)/vseries(px[s])-1).reindex(INDEX) for s in WATCH}).sort_index()

fwd10 = fwd_h(H)
print("ASOF:", ASOF, " n_dates:", len(INDEX))

print("\nGATE abs IC>=0.0070, abs ICIR>=0.0840")

# ---------- flip_mom_20x10 ----------
f = pd.DataFrame({s: flip_mom(px[s]) for s in WATCH}).sort_index().replace([np.inf,-np.inf], np.nan)
icd = cross_sectional_ic(f, fwd10)
st = ic_stats(icd)
def q252(icdf):
    sub = icdf[icdf.index >= icdf.index[-1]-pd.Timedelta(days=365)]
    return ic_stats(sub)
icr = q252(icd)
print("\n=== flip_mom_20x10 ===")
print(f"FULL:  IC={st['ic']:.4f} |IC|={abs(st['ic']):.4f} ICIR={st['icir']:.4f} |ICIR|={abs(st['icir']):.4f} hit={st['hit']:.3f} n={st['n_dates']} avg_n={st['avg_n']:.1f}")
print(f"365d:  IC={icr['ic']:.4f} ICIR={icr['icir']:.4f} hit={icr['hit']:.3f} n={icr['n_dates']}")
for lab, m in [('2020-21', icd.index<pd.Timestamp('2022-01-01')),
               ('2022-23', (icd.index>=pd.Timestamp('2022-01-01'))&(icd.index<pd.Timestamp('2024-01-01'))),
               ('2024-26', (icd.index>=pd.Timestamp('2024-01-01'))&(icd.index<pd.Timestamp('2026-07-01'))),
               ('26H2-27', (icd.index>=pd.Timestamp('2026-07-01'))&(icd.index<pd.Timestamp('2028-01-01'))),
               ('2028', (icd.index>=pd.Timestamp('2028-01-01'))&(icd.index<pd.Timestamp('2029-01-01'))),
               ('2029', (icd.index>=pd.Timestamp('2029-01-01'))&(icd.index<pd.Timestamp('2030-01-01'))),
               ('2030', icd.index>=pd.Timestamp('2030-01-01'))]:
    sub = icd[m]
    if len(sub):
        ss = ic_stats(sub)
        print(f"  regime {lab:12s}: IC={ss['ic']:.4f} ICIR={ss['icir']:.4f} n={ss['n_dates']}")
for hh in [1,5,10,20]:
    icd_h = cross_sectional_ic(f, fwd_h(hh))
    print(f"  decay h={hh}: IC={icd_h['ic'].mean():.4f}" if len(icd_h) else f"  decay h={hh}: NA")

# ---------- usdcny_beta_60 ----------
def usdcny_beta(p, kw=60):
    macro = load_macro(ASOF)
    usdcny = macro['USDCNY']
    out = {}
    for s in WATCH:
        a = vseries(px[s]); b = vseries(usdcny)
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        ra, rb = a.pct_change(), b.pct_change()
        m = ra.notna() & rb.notna()
        if m.sum() < 40:
            out[s] = np.nan; continue
        cov = ra[m].cov(rb[m]); var = rb[m].var()
        out[s] = cov/var if var and var > 0 and not np.isnan(var) else np.nan
    return pd.Series(out)

fb = pd.DataFrame({s: usdcny_beta(px[s]) for s in WATCH}).replace([np.inf,-np.inf], np.nan)
icd_b = cross_sectional_ic(fb, fwd10)
st_b = ic_stats(icd_b); icr_b = q252(icd_b)
print("\n=== usdcny_beta_60 ===")
print(f"FULL:  IC={st_b['ic']:.4f} |IC|={abs(st_b['ic']):.4f} ICIR={st_b['icir']:.4f} |ICIR|={abs(st_b['icir']):.4f} hit={st_b['hit']:.3f} n={st_b['n_dates']} avg_n={st_b['avg_n']:.1f}")
print(f"365d:  IC={icr_b['ic']:.4f} ICIR={icr_b['icir']:.4f} hit={icr_b['hit']:.3f} n={icr_b['n_dates']}")
for lab, m in [('2020-21', icd_b.index<pd.Timestamp('2022-01-01')),
               ('2022-23', (icd_b.index>=pd.Timestamp('2022-01-01'))&(icd_b.index<pd.Timestamp('2024-01-01'))),
               ('2024-26', (icd_b.index>=pd.Timestamp('2024-01-01'))&(icd_b.index<pd.Timestamp('2026-07-01'))),
               ('26H2-27', (icd_b.index>=pd.Timestamp('2026-07-01'))&(icd_b.index<pd.Timestamp('2028-01-01'))),
               ('2028', (icd_b.index>=pd.Timestamp('2028-01-01'))&(icd_b.index<pd.Timestamp('2029-01-01'))),
               ('2029', (icd_b.index>=pd.Timestamp('2029-01-01'))&(icd_b.index<pd.Timestamp('2030-01-01'))),
               ('2030', icd_b.index>=pd.Timestamp('2030-01-01'))]:
    sub = icd_b[m]
    if len(sub):
        ss = ic_stats(sub)
        print(f"  regime {lab:12s}: IC={ss['ic']:.4f} ICIR={ss['icir']:.4f} n={ss['n_dates']}")
