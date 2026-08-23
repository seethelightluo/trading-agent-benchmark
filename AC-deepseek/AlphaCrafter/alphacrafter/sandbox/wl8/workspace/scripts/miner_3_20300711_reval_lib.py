"""miner_3 2030-07-11: re-validate currently effective library factors (flip_mom_20x10, usdcny_beta_60)
through current visible date for drift/timeliness."""
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
print("ASOF:", ASOF)

# ---------- flip_mom_20x10 ----------
f = pd.DataFrame({s: flip_mom(px[s]) for s in WATCH}).sort_index().replace([np.inf,-np.inf], np.nan)
icd = cross_sectional_ic(f, fwd10)
st = ic_stats(icd)
icr = ic_stats(icd[icd.index >= icd.index[-1]-pd.Timedelta(days=252)])
ic60 = ic_stats(icd.tail(60))
print("\n=== flip_mom_20x10 ===")
print(f"FULL: IC={st['ic']:.4f} ICIR={st['icir']:.4f} hit={st['hit']:.3f} n={st['n_dates']} avg_n={st.get('avg_n',float('nan')):.1f}")
print(f"252d: IC={icr['ic']:.4f} ICIR={icr['icir']:.4f} hit={icr['hit']:.3f} n={icr['n_dates']}")
print(f"last60: IC={ic60['ic']:.4f} ICIR={ic60['icir']:.4f} n={ic60['n_dates']}")
for hh in [1,5,10,20]:
    icd_h = cross_sectional_ic(f, fwd_h(hh))
    print(f"  decay h={hh}: IC={icd_h['ic'].mean():.4f}" if len(icd_h) else f"  decay h={hh}: NA")
for lab, m in [('2020-21', icd.index<pd.Timestamp('2022-01-01')),
               ('2022-23', (icd.index>=pd.Timestamp('2022-01-01'))&(icd.index<pd.Timestamp('2024-01-01'))),
               ('2024-26', (icd.index>=pd.Timestamp('2024-01-01'))&(icd.index<pd.Timestamp('2026-07-01'))),
               ('2026-07+', icd.index>=pd.Timestamp('2026-07-01'))]:
    sub = icd[m]
    if len(sub):
        ss = ic_stats(sub)
        print(f"  regime {lab}: IC={ss['ic']:.4f} ICIR={ss['icir']:.4f} n={ss['n_dates']}")

# ---------- usdcny_beta_60 ----------
def usdcny_beta_60(p, kw=60):
    macro = load_macro(ASOF)
    usdcny = macro['USDCNY']
    out = {}
    for s in WATCH:
        a = vseries(px[s])
        b = vseries(usdcny)
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        ra, rb = a.pct_change(), b.pct_change()
        m = ra.notna() & rb.notna()
        if m.sum() < 40:
            out[s] = np.nan
            continue
        cov = ra[m].cov(rb[m]); var = rb[m].var()
        out[s] = cov/var if var and var > 0 and not np.isnan(var) else np.nan
    return pd.Series(out)

fb = pd.DataFrame({s: usdcny_beta_60(px[s]) for s in WATCH}).replace([np.inf,-np.inf], np.nan)
icd_b = cross_sectional_ic(fb, fwd10)
st_b = ic_stats(icd_b)
icr_b = ic_stats(icd_b[icd_b.index >= icd_b.index[-1]-pd.Timedelta(days=252)])
print("\n=== usdcny_beta_60 ===")
print(f"FULL: IC={st_b['ic']:.4f} ICIR={st_b['icir']:.4f} hit={st_b['hit']:.3f} n={st_b['n_dates']} avg_n={st_b.get('avg_n',float('nan')):.1f}")
print(f"252d: IC={icr_b['ic']:.4f} ICIR={icr_b['icir']:.4f} hit={icr_b['hit']:.3f} n={icr_b['n_dates']}")
print(f"last60: IC={ic_stats(icd_b.tail(60))['ic']:.4f}")
for lab, m in [('2020-21', icd_b.index<pd.Timestamp('2022-01-01')),
               ('2022-23', (icd_b.index>=pd.Timestamp('2022-01-01'))&(icd_b.index<pd.Timestamp('2024-01-01'))),
               ('2024-26', (icd_b.index>=pd.Timestamp('2024-01-01'))&(icd_b.index<pd.Timestamp('2026-07-01'))),
               ('2026-07+', icd_b.index>=pd.Timestamp('2026-07-01'))]:
    sub = icd_b[m]
    if len(sub):
        ss = ic_stats(sub)
        print(f"  regime {lab}: IC={ss['ic']:.4f} ICIR={ss['icir']:.4f} n={ss['n_dates']}")

print("\nGATE abs IC>=0.0070, abs ICIR>=0.0840")
print(f"flip_mom: |IC|={abs(st['ic']):.4f} |ICIR|={abs(st['icir']):.4f}")
print(f"usdcny_beta: |IC|={abs(st_b['ic']):.4f} |ICIR|={abs(st_b['icir']):.4f}")