import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(sym,n=4000):
 d=get_stock_daily_data(sym,n)
 if d is None or len(d)==0:return pd.Series(dtype=float)
 return d.set_index(pd.to_datetime(d.date)).close.astype(float).sort_index()
px=pd.DataFrame({s:get(s) for s in U}).sort_index(); ret=px.pct_change()
r10=px.pct_change(10); r60=px.pct_change(60)
down=ret.where(ret<0).rolling(20,min_periods=10).std()*np.sqrt(20)
sig=(.7*r10+.3*r60)/(down+1e-6); fwd=px.shift(-10)/px-1
rows=[];by=[]; ns=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  v=z.iloc[:,0].rank().corr(z.iloc[:,1].rank())
  if np.isfinite(v):rows.append(v);by.append(dt);ns.append(len(z))
ic=np.array(rows); print('dates',len(ic),'mean_n',np.mean(ns),'coverage',len(ic)/len(sig.dropna(how='all')))
print('IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',np.mean(ic>0))
for name,cut in [('2026-07-16',pd.Timestamp('2026-07-16')),('2028-01-01',pd.Timestamp('2028-01-01')),('2029-01-01',pd.Timestamp('2029-01-01'))]:
 a=ic[np.array(by)>=cut];print(name,'n',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
for h in [1,5,10,20]:
 yy=px.shift(-h)/px-1;aa=[]
 for d in sig.index:
  q=pd.concat([sig.loc[d],yy.loc[d]],axis=1).dropna()
  if len(q)>=8:
   v=q.iloc[:,0].rank().corr(q.iloc[:,1].rank())
   if np.isfinite(v):aa.append(v)
 print('h',h,'IC',np.mean(aa),'ICIR',np.mean(aa)/np.std(aa,ddof=1))
