import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=4000)
   if x is not None and len(x): return x
  except Exception: pass
raw={}
for s in U:
 x=fetch(s)
 if x is not None:
  x=x.copy(); x.date=pd.to_datetime(x.date); raw[s]=x.set_index('date').sort_index().close.astype(float)
p=pd.DataFrame(raw).sort_index(); r=p.pct_change()
# Contrarian location in trailing 20-day range, smoothed by volatility: low range location => positive reversal
lo=p.rolling(20,min_periods=15).min(); hi=p.rolling(20,min_periods=15).max()
loc=(p-lo)/(hi-lo).replace(0,np.nan)
vol=r.rolling(20,min_periods=12).std()
f=(0.5-loc)/vol

def eval_h(h):
 vals=[]; ns=[]
 fr=r.shift(-1).rolling(h).sum().shift(-(h-1))
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>=3:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 q=pd.Series(vals).dropna(); return q,ns
q,ns=eval_h(1)
turn=[]; prev=None
for dt in f.index:
 z=f.loc[dt].dropna()
 if len(z)>=8:
  rk=z.rank(pct=True)
  if prev is not None: turn.append(abs(rk-prev.reindex(rk.index)).mean())
  prev=rk
print('assets',len(raw),'dates',len(q),'avg_instruments',round(np.mean(ns),2),'coverage',round(np.mean(ns)/len(U),4),'turnover',round(np.mean(turn),5))
print('horizon 1 IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
for h in [5,10]:
 s,n=eval_h(h); print('horizon',h,'dates',len(s),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6))
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31')]:
 v=q[(q.index.astype(str)>=a)&(q.index.astype(str)<=b)]; print('regime',a,'n',len(v),'IC',round(v.mean(),6) if len(v) else None)
