import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 z=get_stock_daily_data(s,days=3000)
 if z is not None and len(z): P[s]=z.assign(date=pd.to_datetime(z.date)).set_index('date').close.astype(float)
dates=sorted(set.intersection(*[set(x.index) for x in P.values()]))
for L in [10,15,20,30,40]:
 rows=[]
 for i,d in enumerate(dates):
  if i<L+2 or i+10>=len(dates): continue
  f={}; y={}
  for s,x in P.items():
   try:
    r=x.pct_change().loc[:d].tail(L).dropna()
    f[s]=(x.loc[d]/x.loc[dates[i-L]]-1)*(0.5+0.5*(r>0).mean()); y[s]=x.loc[dates[i+10]]/x.loc[d]-1
   except: pass
  if len(f)>=8:
   c=pd.Series(f).corr(pd.Series(y),method='spearman')
   if np.isfinite(c): rows.append(c)
 q=np.array(rows); print('lookback',L,'dates',len(q),'avgN',len(P),'coverage',len(P)/15,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
