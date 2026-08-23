import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   d=f(s,days=4000)
   if d is not None:return d
  except Exception: pass
raw={}
for s in U:
 d=fetch(s)
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); raw[s]=d.set_index('date').sort_index().close.astype(float)
p=pd.DataFrame(raw).sort_index(); r=p.pct_change()
recent=r.rolling(10,min_periods=8).sum(); prior=r.shift(10).rolling(30,min_periods=20).sum()
vol=r.rolling(40,min_periods=25).std(); f=(recent-prior)/vol
ics=[]; ns=[]; turns=[]; prev=None
for dt in f.index:
 z=pd.concat([f.loc[dt],r.shift(-1).loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>=3:
  ics.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))); ns.append(len(z))
  rk=z.iloc[:,0].rank(pct=True)
  if prev is not None: turns.append(abs(rk-prev.reindex(rk.index)).mean())
  prev=rk
q=pd.Series(dict(ics)).dropna(); print('assets',len(raw),'dates',len(q),'avg_instruments',round(np.mean(ns),2),'coverage',round(np.mean(ns)/len(U),4),'turnover',round(np.mean(turns),5))
print('horizon 1 IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
for h in [5,10]:
 vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],r.shift(-1).rolling(h).sum().shift(-(h-1)).loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 s=pd.Series(vals).dropna(); print('horizon',h,'dates',len(s),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6))
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31')]:
 v=q[(q.index.astype(str)>=a)&(q.index.astype(str)<=b)]; print('regime',a,'n',len(v),'IC',round(v.mean(),6) if len(v) else None)
