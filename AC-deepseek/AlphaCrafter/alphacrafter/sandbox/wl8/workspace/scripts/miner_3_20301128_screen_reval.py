import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from miner_3_20261203_common import WATCH, load_prices, load_macro, load_visible_through, cross_sectional_ic, ic_stats, spearman_panel_rho

ASOF = load_visible_through(); H=10
px = load_prices(ASOF); INDEX=px.index
mac = load_macro(ASOF)
def v(s): return s.dropna()
def retk(s,k):
    vv=v(s); return (vv/vv.shift(k)-1.0).reindex(INDEX)
def flip_mom(p,kw=20,ks=10):
    return (retk(p,kw)*np.sign(retk(p,ks))).reindex(INDEX)
def fwd_h(h):
    return pd.DataFrame({s:(v(px[s]).shift(-h)/v(px[s])-1).reindex(INDEX) for s in WATCH}).sort_index()
fwd10 = fwd_h(H)
print("ASOF", ASOF, "px_last", px.index[-1].date())
print("frozen60:", [s for s in WATCH if len(v(px[s]).tail(60))>0 and v(px[s]).tail(60).std()<1e-12])

usc = retk(mac['USDCNY'],1)
def usdcny_beta(p, w=60):
    ra = retk(p,1)
    out = pd.Series(np.nan, index=INDEX)
    for i in range(w, len(INDEX)):
        a = ra.iloc[i-w:i].values; b = usc.iloc[i-w:i].values
        m = np.isfinite(a)&np.isfinite(b)
        if m.sum() >= 30:
            A=a[m]; B=b[m]; va=B.var(ddof=1)
            if va>0:
                out.iat[i]=float(np.cov(A,B,ddof=1)[0,1]/va)
    return out
FB = pd.DataFrame({s:flip_mom(px[s]) for s in WATCH}).sort_index().replace([np.inf,-np.inf],np.nan)
UB = pd.DataFrame({s:usdcny_beta(px[s]) for s in WATCH}).sort_index().replace([np.inf,-np.inf],np.nan)

def report(name, f):
    icd = cross_sectional_ic(f, fwd10)
    st=ic_stats(icd); icr=ic_stats(icd[icd.index>=icd.index[-1]-pd.Timedelta(days=252)]); ic60=ic_stats(icd.tail(60)); ic180=ic_stats(icd.tail(180))
    print(f"\n=== {name} ===")
    print(f"FULL IC={st['ic']:.4f} ICIR={st['icir']:.4f} hit={st['hit']:.3f} n={st['n_dates']} (gate {'PASS' if abs(st['ic'])>=0.007 and abs(st['icir'])>=0.084 else 'FAIL'})")
    print(f"252d IC={icr['ic']:.4f} ICIR={icr['icir']:.4f} n={icr['n_dates']}")
    print(f"180d IC={ic180['ic']:.4f} ICIR={ic180['icir']:.4f} n={ic180['n_dates']}")
    print(f"60d  IC={ic60['ic']:.4f} ICIR={ic60['icir']:.4f} n={ic60['n_dates']}")
    for hh in [1,5,10,20]:
        ih=cross_sectional_ic(f,fwd_h(hh)); print(f"  decay h={hh}: IC={ih['ic'].mean():.4f}" if len(ih) else f"  decay h={hh}: NA")
    for lab,m in [('2020-21',icd.index<pd.Timestamp('2022-01-01')),('2022-23',(icd.index>=pd.Timestamp('2022-01-01'))&(icd.index<pd.Timestamp('2024-01-01'))),('2024-26',(icd.index>=pd.Timestamp('2024-01-01'))&(icd.index<pd.Timestamp('2026-07-01'))),('2026+',icd.index>=pd.Timestamp('2026-07-01'))]:
        sub=icd[m]
        if len(sub):
            ss=ic_stats(sub); print(f"  regime {lab}: IC={ss['ic']:.4f} ICIR={ss['icir']:.4f} n={ss['n_dates']}")
report("flip_mom_20x10", FB)
report("usdcny_beta_60", UB)
print("\ncorr flip_mom vs usdcny_beta (datewise avg rho):", spearman_panel_rho(FB, UB))
for name,f in [('flip_mom',FB),('usdcny',UB)]:
    icd=cross_sectional_ic(f,fwd10)
    for lab,delta in [('3m',90),('6m',180),('1y',252)]:
        sub=icd[icd.index>=icd.index[-1]-pd.Timedelta(days=delta)]
        if len(sub):
            ss=ic_stats(sub); print(f"{name} {lab}: IC={ss['ic']:.4f} ICIR={ss['icir']:.4f} n={ss['n_dates']}")