import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   x=f(s,days=5000)
   if x is not None and len(x): return x
  except: pass
D={s:get(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
px=pd.concat({s:x.set_index(pd.to_datetime(x.date)).close for s,x in D.items()},axis=1).sort_index().ffill()
# low-volatility with positive medium-term trend: inverse 60d vol, neutralized by cross-sectional trend sign
vol=px.pct_change().rolling(60).std(); trend=px.pct_change(120)
fac=(-vol).where(trend>0, -vol*0.25)
for h in [5,10,20]:
 fwd=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank())); ns.append(len(z))
 q=pd.Series(vals).dropna(); print(h,len(q),np.mean(ns),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),fac.notna().mean().mean())
print('cutoff',px.index.max().date())
