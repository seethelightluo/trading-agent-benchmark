import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for a in A:
 try:D[a]=get_stock_daily_data(a,days=2000)
 except Exception:
  try:D[a]=get_index_daily_data(a,days=2000)
  except:D[a]=None
P=pd.concat([d.set_index('date').close.astype(float).rename(a) for a,d in D.items() if d is not None and len(d)>100],axis=1).sort_index(); R=P.pct_change()
for w in [10,20,60]:
 qs=[]
 for i in range(w+2,len(P)-1):
  x=R.iloc[i-w:i].std(); y=R.iloc[i+1]/1 # next return aligned instruments
  z=pd.concat([(-x).rename('x'),y.rename('y')],axis=1).dropna()
  if len(z)>=8: qs.append(z.x.corr(z.y))
 q=pd.Series(qs).dropna(); print('w',w,'dates',len(q),'N',len(P.columns),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
print('coverage',R.rolling(60).std().notna().sum(axis=1).ge(8).mean(),'period',P.index.min(),P.index.max())
