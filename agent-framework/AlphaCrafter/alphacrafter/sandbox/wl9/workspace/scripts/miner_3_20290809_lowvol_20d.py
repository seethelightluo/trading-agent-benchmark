import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 d=get_stock_daily_data(s,4000)
 return d.set_index(pd.to_datetime(d.date)).close.astype(float).sort_index() if d is not None and len(d) else pd.Series(dtype=float)
p=pd.concat([g(s).rename(s) for s in U],axis=1).sort_index().ffill(); r=p.pct_change(); f=-(r.rolling(20,min_periods=15).std()*np.sqrt(20)); y=p.shift(-10)/p-1
for st in [None,'2026-07-16','2028-01-01','2029-01-01']:
 a=[];ns=[]
 for d in f.index:
  if st and d<pd.Timestamp(st):continue
  q=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8:
   v=q.iloc[:,0].rank().corr(q.iloc[:,1].rank())
   if np.isfinite(v):a.append(v);ns.append(len(q))
 a=np.array(a); print(st or 'full','n',len(a),'mean_n',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
