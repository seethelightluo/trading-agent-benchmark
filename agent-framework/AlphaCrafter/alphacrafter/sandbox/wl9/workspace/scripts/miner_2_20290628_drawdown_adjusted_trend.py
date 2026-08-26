import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s,n=4000):
 d=get_stock_daily_data(s,n)
 if d is None or len(d)==0:return pd.Series(dtype=float)
 return d.set_index(pd.to_datetime(d.date)).close.astype(float).sort_index()
px=pd.DataFrame({s:get(s) for s in U}).sort_index(); ret=px.pct_change()
r20=px.pct_change(20); r60=px.pct_change(60)
down=ret.where(ret<0).rolling(30,min_periods=15).std()*np.sqrt(30)
dd=px/px.rolling(120,min_periods=60).max()-1
sig=(0.65*r20+0.35*r60)/(down+1e-6) * (1+0.5*dd.clip(-1,0))
rows=[]; dates=[]; ns=[]
for d in sig.index:
 z=pd.concat([sig.loc[d],(px.shift(-10)/px-1).loc[d]],axis=1).dropna()
 if len(z)>=8:
  v=z.iloc[:,0].rank().corr(z.iloc[:,1].rank())
  if np.isfinite(v): rows.append(v);dates.append(pd.Timestamp(d));ns.append(len(z))
a=np.array(rows); dates=np.array(dates,dtype='datetime64[ns]')
print('dates',len(a),'mean_n',np.mean(ns),'coverage',len(a)/len(sig.dropna(how='all')))
print('IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
for label,cut in [('online','2026-07-16'),('recent252','2028-06-28'),('2029','2029-01-01')]:
 q=a[dates>=np.datetime64(cut)]; print(label,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',np.mean(q>0))
for h in [5,10,20]:
 aa=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],(px.shift(-h)/px-1).loc[d]],axis=1).dropna()
  if len(z)>=8:
   v=z.iloc[:,0].rank().corr(z.iloc[:,1].rank())
   if np.isfinite(v):aa.append(v)
 aa=np.array(aa); print('h',h,'n',len(aa),'IC',aa.mean(),'ICIR',aa.mean()/aa.std(ddof=1))
