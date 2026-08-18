import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').sort_index().close.astype(float).pct_change()
r=pd.DataFrame(D).sort_index()
ret20=r.rolling(20).sum(); vol=r.rolling(20).std(); persist=r.gt(0).rolling(20).mean()
f=(ret20/(vol*np.sqrt(20)+1e-12)*(0.5+persist)).shift(1)
fr=r.shift(-1).rolling(10).sum().shift(-9)
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(q),'avg_n',q.n.mean(),'coverage',q.n.sum()/(len(q)*15))
print('IC10 %.6f ICIR %.6f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean()))
for h in [1,5,10,20]:
 ff=r.shift(-1).rolling(h).sum().shift(-(h-1)); rr=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('decay',h,np.nanmean(rr),'n',len(rr))
for a,b in [('2026-07-16','2028-12-31'),('2029-01-01','2032-12-31'),('2033-01-01','2035-10-26')]:
 z=q.loc[a:b,'ic']; print('regime',a,b,len(z),z.mean(),z.mean()/z.std(ddof=1))
rank=f.rank(axis=1,pct=True); print('turnover',((rank-rank.shift(1)).abs().mean(axis=1)).mean())
