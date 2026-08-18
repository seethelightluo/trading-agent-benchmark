import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x):
  x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').sort_index().close.astype(float).pct_change()
r=pd.DataFrame(D).sort_index(); v=r.rolling(20).std(); ret=r.rolling(20).sum()
# oversold reversal, strengthened when recent vol is above its 60d median
shock=(v/v.rolling(60).median()).clip(0.5,2.0)
f=(-ret/(v*np.sqrt(20)+1e-12)*shock).shift(1)
fr=r.shift(-1).rolling(10).sum().shift(-9); rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('dates',len(q),'avg_n',q.n.mean(),'coverage',q.n.sum()/(len(q)*15));print('IC10 %.6f ICIR %.6f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean()))
for a,b in [('2026-07-16','2028-12-31'),('2029-01-01','2032-12-31'),('2033-01-01','2035-10-26')]:
 z=q.loc[a:b,'ic'];print('regime',a,b,len(z),z.mean(),z.mean()/z.std(ddof=1))
print('turnover',((f.rank(axis=1,pct=True)-f.rank(axis=1,pct=True).shift(1)).abs().mean(axis=1)).mean())
