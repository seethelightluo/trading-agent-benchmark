"""miner_1 2031-12-11: deep-dive vol-family + fresh candidates. Resolve sign ambiguity
between abs_ret_20 (neg) and rv20. Test windows, sign consistency, recency, correlation
with fallback ensemble, and confirm no sign flip across firms vs dates.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from miner_3_20261203_common import (WATCH, load_prices, load_macro, load_visible_through,
                                     cross_sectional_ic, ic_stats, regime_split, spearman_panel_rho)

ASOF = load_visible_through()
px = load_prices(ASOF); mac = load_macro(ASOF)
INDEX = px.index
print(f"ASOF={ASOF} rows={len(INDEX)} assets={len(WATCH)}")

def vseries(s): return s.dropna()
def retk(s,k):
    v=vseries(s); return (v/v.shift(k)-1.0).reindex(INDEX)
def forward(s,h):
    v=vseries(s); return (v.shift(-h)/v-1.0).reindex(INDEX)
def rv(s,w):
    v=vseries(s); return v.pct_change().rolling(w).std().reindex(INDEX)
def abs_ret(s,w):
    v=vseries(s); return v.pct_change().abs().rolling(w).mean().reindex(INDEX)

H=10
fwd=pd.DataFrame({s:forward(px[s],H) for s in WATCH}).sort_index()
def build(df): return df.sort_index().replace([np.inf,-np.inf],np.nan).astype(float)
def assess(name,factor_df):
    icd=cross_sectional_ic(factor_df,fwd)
    if len(icd)==0: print(f"{name:30s} NO DATES"); return None
    st=ic_stats(icd)
    line=f"{name:30s} IC={st['ic']:+.4f} ICIR={st['icir']:+.4f} hit={st['hit']:.3f} n={st['n_dates']}"
    rmask=icd.index>=icd.index[-1]-pd.Timedelta(days=365)
    if rmask.any():
        ic365=ic_stats(icd[rmask]); line+=f" | 365d {ic365['ic']:+.4f}/{ic365['icir']:+.4f}"
    g=abs(st['ic'])>=0.0070 and abs(st['icir'])>=0.0840
    line+=f" | {'PASS' if g else 'FAIL'}"
    print(line)
    return st

print("===== VOL FAMILY (std) =====")
for w in [10,20,30,60]:
    f=build(pd.DataFrame({s:rv(px[s],w) for s in WATCH}))
    assess(f'rv{w} (pos)', f)
    assess(f'rv{w} (neg)', -f)

print("\n===== VOL FAMILY (abs) =====")
for w in [10,20,30,60]:
    f=build(pd.DataFrame({s:abs_ret(px[s],w) for s in WATCH}))
    assess(f'abs_ret_{w} (pos)', f)
    assess(f'abs_ret_{w} (neg)', -f)

# correlation between rv20 and abs_ret20 signals
f_rv20=build(pd.DataFrame({s:rv(px[s],20) for s in WATCH}))
f_ab20=build(pd.DataFrame({s:abs_ret(px[s],20) for s in WATCH}))
print("\npanel rho rv20 vs abs_ret20:", round(spearman_panel_rho(f_rv20,f_ab20),4))

# correlation with fallback mom10 and vix-beta
f_mom=build(pd.DataFrame({s:retk(px[s],10) for s in WATCH}))
def beta_cond(p,reg,window=60,cond=20):
    a=retk(p,1); b=retk(reg,1); mb=retk(reg,cond).reindex(INDEX)
    beta=a.rolling(window).cov(b)/b.rolling(window).var()
    return build(beta*np.sign(mb))
f_vb=build(pd.DataFrame({s:beta_cond(px[s],mac['VIX']) for s in WATCH}))
print("rho rv20 vs mom10:", round(spearman_panel_rho(f_rv20,f_mom),4))
print("rho rv20 vs vix_beta:", round(spearman_panel_rho(f_rv20,f_vb),4))
print("rho abs_ret20 vs vix_beta:", round(spearman_panel_rho(f_ab20,f_vb),4))

print("\nDONE")