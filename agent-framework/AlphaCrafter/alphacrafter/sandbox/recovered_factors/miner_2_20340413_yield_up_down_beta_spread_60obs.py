"""Miner_2: conditional cross-asset sensitivity to global yield shocks; completed bars only."""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=list(get_account_dict()['watch_list'])
def returns(a):
    d=get_stock_daily_data(a,days=3800)
    return pd.to_numeric(d['close'],errors='coerce').pct_change().replace([np.inf,-np.inf],np.nan)
r=pd.DataFrame({a:returns(a) for a in A})
# Yield assets are themselves tradable; their median change is solely an observed cross-asset state input.
y=r[['US10Y','CN10Y']].median(axis=1)
def beta(a,b):
 z=pd.concat([a,b],axis=1).dropna()
 if len(z)<8:return np.nan
 v=z.iloc[:,1].var()
 return z.iloc[:,0].cov(z.iloc[:,1])/v if v>1e-14 else np.nan
# The signal measures whether exposure to upward yield shocks differs from exposure to downward shocks.
# Every trailing window ends t-1, so signal at t is decision-time feasible.
sig=pd.DataFrame(np.nan,index=r.index,columns=A)
for p in range(60,len(r)):
 x=y.iloc[p-60:p]; up=x>=x.median()
 for a in A:
  sig.iloc[p,sig.columns.get_loc(a)]=beta(r[a].iloc[p-60:p][up],x[up])-beta(r[a].iloc[p-60:p][~up],x[~up])
print('CANDIDATE yield_up_vs_down_beta_spread_60obs')
print('dates',len(r),'range',r.index.min(),r.index.max(),'signal_cells',int(sig.notna().sum().sum()),'/',sig.size)
def stats(z):
 return (z.mean(),z.mean()/z.std(ddof=1) if z.std(ddof=1)>0 else np.nan,(z>0).mean())
for h in (1,5,10,20):
 vals=[]; ns=[]
 for p in range(60,len(r)-h):
  f=sig.iloc[p]; fw=(1+r.iloc[p+1:p+1+h]).prod()-1; ok=f.notna()&fw.notna()
  if ok.sum()>=8: vals.append(f[ok].corr(fw[ok],method='spearman'));ns.append(ok.sum())
 z=np.array(vals); q=stats(z)
 print('H',h,'IC',round(q[0],6),'ICIR',round(q[1],6),'hit',round(q[2],4),'dates',len(z),'meanN',round(np.mean(ns),2))
 for nm,sub in zip(('early','middle','recent'),np.array_split(z,3)):
  w=stats(sub);print(' ',nm,'IC',round(w[0],6),'ICIR',round(w[1],6),'N',len(sub))
t=[]
for p in range(61,len(sig)):
 x=sig.iloc[p-1];z=sig.iloc[p];ok=x.notna()&z.notna()
 if ok.sum()>=8:t.append((x[ok].rank(pct=True)-z[ok].rank(pct=True)).abs().mean())
print('turnover',round(float(np.mean(t)),6),'comparisons',len(t),'coverage',round(float(sig.notna().mean().mean()),4))
# Save raw signal only for a subsequent required library-correlation audit if predictive gates pass.
sig.to_pickle('scripts/miner_2_yield_asymmetry_candidate_signal.pkl')
