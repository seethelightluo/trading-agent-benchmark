import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-07-15')
D={}
for s in U:
 x=get_stock_daily_data(s,days=3000)
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date).dt.normalize(); x=x[x.date<=cut].drop_duplicates('date').set_index('date').sort_index(); D[s]=x.close.astype(float)
P=pd.concat(D,axis=1).sort_index(); R=P.pct_change(fill_method=None)
# volatility-adjusted short reversal, signal known at t, forward t+1
for w in [3,5,10]:
 fac=-R.rolling(w,min_periods=w).sum()/R.rolling(20,min_periods=15).std()
 rows=[]
 for dt in fac.index[:-1]:
  z=pd.concat([fac.loc[dt],R.shift(-1).loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1])))
 g=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); ic=g.ic.replace([np.inf,-np.inf],np.nan).dropna()
 print('w',w,'dates',len(ic),'avg_n',g.n.mean(),'coverage',g.n.sum()/(len(ic)*15),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean(),'turnover',np.nan)
 for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-15')]:
  q=ic[(ic.index>=lo)&(ic.index<=hi)]; print(' regime',lo,len(q),q.mean(),q.mean()/q.std(ddof=1))
 for h in [5,10]:
  yy=R.shift(-1).rolling(h,min_periods=h).sum().shift(-(h-1)); vals=[]
  for dt in fac.index:
   z=pd.concat([fac.loc[dt],yy.loc[dt]],axis=1).dropna()
   if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
  q=pd.Series(vals).dropna(); print(' horizon',h,len(q),q.mean(),q.mean()/q.std(ddof=1))
print('data',P.index.min(),P.index.max(),P.shape)
