"""miner_3 2030-10-30: re-validate currently effective library factors
(flip_mom_20x10, usdcny_beta_60) through current visible date (2030-10-30)
for drift/timeliness, plus regime snapshot."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, json
from miner_3_20261203_common import WATCH, load_prices, load_macro, load_visible_through, cross_sectional_ic, ic_stats

ASOF = load_visible_through()
H = 10
px = load_prices(ASOF)
INDEX = px.index
mac = load_macro(ASOF)
print("ASOF visible_through:", ASOF)
print("px shape:", px.shape)

def vseries(s): return s.dropna()
def retk(s, k):
    v = vseries(s)
    return (v / v.shift(k) - 1.0).reindex(INDEX)

def flip_mom(p, kw=20, ks=10):
    return (retk(p, kw) * np.sign(retk(p, ks))).reindex(INDEX)

def fwd_h(h):
    return pd.DataFrame({s: (vseries(px[s]).shift(-h)/vseries(px[s])-1).reindex(INDEX) for s in WATCH}).sort_index()

fwd10 = fwd_h(H)

# ---------- regime snapshot ----------
print("\n=== regime snapshot (25d / 60d returns) ===")
for s in WATCH:
    v = vseries(px[s])
    if len(v) < 70:
        print(f"{s:10s} insufficient data {len(v)}")
        continue
    r25 = v.iloc[-1]/v.iloc[-26]-1
    r60 = v.iloc[-1]/v.iloc[-61]-1
    print(f"{s:10s} last={v.iloc[-1]:12.2f} 25d={r25*100:8.2f}% 60d={r60*100:8.2f}%")
print("\nmacro last levels:")
for m in mac.columns:
    v = mac[m].dropna()
    if len(v) >= 26:
        print(f"{m:8s} {v.iloc[-1]:9.3f} 25d={ (v.iloc[-1]/v.iloc[-26]-1)*100:7.2f}%")
print("\nfrozen (std<1e-12 last 60d):", [s for s in WATCH if len(vseries(px[s]).tail(60)) and vseries(px[s]).tail(60).std() < 1e-12])

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
print(f"GATE: |IC|={abs(st['ic']):.4f} |ICIR|={abs(st['icir']):.4f} -> {'PASS' if abs(st['ic'])>=0.0070 and abs(st['icir'])>=0.0840 else 'FAIL'}")