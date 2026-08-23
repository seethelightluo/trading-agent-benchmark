"""miner_3 2030-05-02: re-validate flip_mom_20x10 through current visible date for drift/timeliness."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, json
from miner_3_20261203_common import WATCH, load_prices, load_macro, cross_sectional_ic, ic_stats

ASOF = '2030-05-01'
H = 10
px = load_prices(ASOF)
INDEX = px.index

def vseries(s): return s.dropna()
def retk(s, k):
    v = vseries(s)
    return (v / v.shift(k) - 1.0).reindex(INDEX)

def flip_mom(p, kw=20, ks=10):
    return (retk(p, kw) * np.sign(retk(p, ks))).reindex(INDEX)

f = pd.DataFrame({s: flip_mom(px[s]) for s in WATCH}).sort_index().replace([np.inf,-np.inf], np.nan)
fwd = pd.DataFrame({s: (vseries(px[s]).shift(-H)/vseries(px[s])-1).reindex(INDEX) for s in WATCH}).sort_index()

icd = cross_sectional_ic(f, fwd)
st = ic_stats(icd)
icr = ic_stats(icd[icd.index >= icd.index[-1]-pd.Timedelta(days=252)])
print(f"FULL: IC={st['ic']:.4f} ICIR={st['icir']:.4f} hit={st['hit']:.3f} n={st['n_dates']} avg_n={st.get('avg_n',float('nan')):.1f}")
print(f"252d: IC={icr['ic']:.4f} ICIR={icr['icir']:.4f} hit={icr['hit']:.3f} n={icr['n_dates']}")

# recent drift: last 60 IC dates
ic60 = ic_stats(icd.tail(60))
print(f"last60d: IC={ic60['ic']:.4f} ICIR={ic60['icir']:.4f} n={ic60['n_dates']}")

# decay up to current
for hh in [1,5,10,20]:
    fh = pd.DataFrame({s: (vseries(px[s]).shift(-hh)/vseries(px[s])-1).reindex(INDEX) for s in WATCH}).sort_index()
    icd_h = cross_sectional_ic(f, fh)
    print(f"  decay h={hh}: IC={icd_h['ic'].mean():.4f}" if len(icd_h) else f"  decay h={hh}: NA")

# regime
for lab, m in [('2020-21', icd.index<pd.Timestamp('2022-01-01')),
               ('2022-23', (icd.index>=pd.Timestamp('2022-01-01'))&(icd.index<pd.Timestamp('2024-01-01'))),
               ('2024-26', (icd.index>=pd.Timestamp('2024-01-01'))&(icd.index<pd.Timestamp('2026-07-01'))),
               ('2026-07+', icd.index>=pd.Timestamp('2026-07-01'))]:
    sub = icd[m]
    if len(sub):
        ss = ic_stats(sub)
        print(f"  regime {lab}: IC={ss['ic']:.4f} ICIR={ss['icir']:.4f} n={ss['n_dates']}")

# significance of recent sign vs admission
print("\nGATE: IC>=", 0.0070, "->", abs(st['ic']), "ICIR>=",0.0840,"->", abs(st['icir']))