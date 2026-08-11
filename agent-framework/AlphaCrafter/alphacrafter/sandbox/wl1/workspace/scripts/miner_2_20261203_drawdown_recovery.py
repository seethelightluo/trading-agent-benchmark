import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-07-15')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut] for s in U}
idx=sorted(set().union(*[set(v.index) for v in P.values()])); p=pd.DataFrame({s:v.reindex(idx) for s,v in P.items()}).ffill()
# Drawdown recovery / distance from long-term high.  A controlled recovery signal
# rewards assets that have recovered from a 120d low while avoiding fully extended highs.
rollmin=p.rolling(120,min_periods=80).min(); rollmax=p.rolling(120,min_periods=80).max()
location=(p-rollmin)/(rollmax-rollmin).replace(0,np.nan)
recovery=p/p.shift(20)-1
f=(recovery*(1-location)).shift(1)
# cross-sectional clipping only; no future information
f=f.clip(f.quantile(.05,axis=1),f.quantile(.95,axis=1),axis=0)
print('universe',len(U),'dates',len(p),'cutoff',p.index.max().date())
for h in [5,10,20]:
 I=[]; Ns=[]; ds=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z): I.append(z); Ns.append(len(q)); ds.append(p.index[i])
 a=np.asarray(I); print('h',h,'valid_dates',len(a),'avgN',round(np.mean(Ns),2),'coverage',round(np.mean(Ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 print('annual',h,{y:round(a[[d.year==y for d in ds]].mean(),6) for y in sorted(set(d.year for d in ds))})
rank=f.rank(axis=1,pct=True); turn=(rank-rank.shift(1)).abs().stack().groupby(level=0).mean().dropna(); print('turnover',round(turn.mean(),6),'coverage_dates',int(f.notna().sum(axis=1).ge(8).sum()))
