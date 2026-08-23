import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   x=f(s,days=5000)
   if x is not None and len(x):return x
  except:pass
D={s:g(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
p=pd.concat({s:x.set_index(pd.to_datetime(x.date)).close for s,x in D.items()},axis=1).sort_index().ffill(); r=p.pct_change()
# cross-asset beta residual over 60d, then contrarian 20d residual standardized by residual vol
b=r.rolling(60).cov(r.mean(axis=1)).div(r.mean(axis=1).rolling(60).var(),axis=0)
res=r.sub(b.mul(r.mean(axis=1),axis=0)); fac=-res.rolling(20).sum()/res.rolling(40).std()
for h in [5,10,20]:
 fw=p.shift(-h)/p-1; z=[]; ns=[]
 for d in fac.index:
  a=pd.concat([fac.loc[d],fw.loc[d]],axis=1).dropna()
  if len(a)>=8 and a.iloc[:,0].nunique()>1 and a.iloc[:,1].nunique()>1:z.append(a.iloc[:,0].rank().corr(a.iloc[:,1].rank()));ns.append(len(a))
 q=pd.Series(z).dropna();print(h,len(q),np.mean(ns),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),fac.notna().mean().mean())
print('cutoff',p.index.max().date())
