import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def f(s):
 for g in (get_index_daily_data,get_stock_daily_data):
  try:
   x=g(s,days=4000)
   if x is not None and len(x): return x
  except: pass
P={}
for s in U:
 x=f(s)
 if x is not None: P[s]=x.set_index(pd.to_datetime(x.date)).close.astype(float)
p=pd.DataFrame(P).sort_index(); r=p.pct_change(); r10=p.pct_change(10); v=r.rolling(20).std()*np.sqrt(20); d=r.std(axis=1).rolling(20).mean(); m=(d/d.rolling(120).median()).clip(.5,2)
s=(-r10.sub(r10.mean(axis=1),axis=0)/(v*np.sqrt(20))).mul(1/m,axis=0).shift(1)
for h in [10,20,40]:
 q=[]
 for t in s.index:
  z=pd.concat([s.loc[t],(p.shift(-h)/p-1).loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1])
   if np.isfinite(c):q.append(c)
 a=np.array(q);print(h,len(a),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean())
