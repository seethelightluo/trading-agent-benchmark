import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0:return pd.Series(dtype=float)
 return d.set_index(pd.to_datetime(d.date)).close.astype(float).sort_index()
px=pd.concat([get(s).rename(s) for s in U],axis=1).sort_index().ffill()
r=px.pct_change(); vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
sig=-(px.pct_change(20)/(vol+1e-8)); fwd=px.shift(-10)/px-1

def calc(y, start=None):
 out=[]; ds=[]; ns=[]
 for d in sig.index:
  if start is not None and d<pd.Timestamp(start): continue
  q=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8:
   x=q.iloc[:,0].rank(); z=q.iloc[:,1].rank(); v=x.corr(z)
   if np.isfinite(v):out.append(v);ds.append(d);ns.append(len(q))
 a=np.array(out); return len(a),np.mean(ns),np.mean(a),np.mean(a)/np.std(a,ddof=1),np.mean(a>0),ds
print('signal', 'negative 20d return / 20d realized volatility')
print('valid dates',len(sig.dropna(how='all')),'full',calc(fwd)[:5])
for st in ['2026-07-16','2028-01-01','2029-01-01']:
 print(st,calc(fwd,st)[:5])
for h in [1,5,10,20,40]:
 n,nn,ic,ir,hit,_=calc(px.shift(-h)/px-1); print('h',h,'n',n,'IC',ic,'ICIR',ir,'hit',hit)
