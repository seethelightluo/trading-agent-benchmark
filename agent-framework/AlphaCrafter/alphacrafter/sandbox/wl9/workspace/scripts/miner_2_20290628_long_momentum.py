import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 d=get_stock_daily_data(s,4000)
 return pd.Series(dtype=float) if d is None else d.set_index(pd.to_datetime(d.date)).close.astype(float).sort_index()
p=pd.DataFrame({s:g(s) for s in U}).sort_index(); sig=p.shift(20)/p.shift(120)-1; rows=[];ds=[];ns=[]
for d in sig.index:
 z=pd.concat([sig.loc[d],(p.shift(-10)/p-1).loc[d]],axis=1).dropna()
 if len(z)>=8:
  x=z.iloc[:,0].rank().corr(z.iloc[:,1].rank())
  if np.isfinite(x):rows.append(x);ds.append(pd.Timestamp(d));ns.append(len(z))
a=np.array(rows); ds=np.array(ds,dtype='datetime64[ns]');print('dates',len(a),'mean_n',np.mean(ns),'coverage',len(a)/len(sig.dropna(how='all')));print('IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
for l,c in [('online','2026-07-16'),('recent','2028-06-28'),('2029','2029-01-01')]:
 q=a[ds>=np.datetime64(c)];print(l,len(q),q.mean(),q.mean()/q.std(ddof=1),np.mean(q>0))
for h in [5,10,20]:
 q=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],(p.shift(-h)/p-1).loc[d]],axis=1).dropna()
  if len(z)>=8:
   x=z.iloc[:,0].rank().corr(z.iloc[:,1].rank())
   if np.isfinite(x):q.append(x)
 q=np.array(q);print('h',h,len(q),q.mean(),q.mean()/q.std(ddof=1))
