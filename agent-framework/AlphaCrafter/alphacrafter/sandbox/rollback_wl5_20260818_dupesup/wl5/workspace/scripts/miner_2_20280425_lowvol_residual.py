import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
a={}
for s in U:
 d=get_stock_daily_data(s,days=2200)
 if d is not None:
  d=d.copy();d.date=pd.to_datetime(d.date);a[s]=d.set_index('date').close.astype(float)
p=pd.concat(a,axis=1).sort_index().ffill(); r=p.pct_change()
# Low-volatility signal, residualized cross-sectionally against trailing momentum to reduce overlap.
v=r.rolling(30).std(); mom=p/p.shift(30)-1
sig=-v
# daily cross-sectional residual of low vol after linear projection on momentum
for dt in sig.index:
 x=pd.concat([sig.loc[dt],mom.loc[dt]],axis=1).dropna()
 if len(x)>=8:
  X=np.c_[np.ones(len(x)),x.iloc[:,1].values]; beta=np.linalg.lstsq(X,x.iloc[:,0].values,rcond=None)[0]
  sig.loc[dt,x.index]=x.iloc[:,0]-X@beta
rows=[]
for dt in sig.index:
 for h in [5,10,20]:
  z=pd.concat([sig.loc[dt],p.shift(-h).loc[dt]/p.loc[dt]-1],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,z.iloc[:,0].rank().corr(z.iloc[:,1].rank()),len(z)))
r=pd.DataFrame(rows,columns=['date','h','ic','n']);print('assets',len(a),'dates',p.index.min(),p.index.max(),'rows',len(r))
for h in [5,10,20]:
 q=r[r.h==h];print('H',h,'obs',len(q),'N',q.n.mean(),'IC %.6f ICIR %.6f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean()))
 for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2099')]:
  z=q[(q.date>=lo)&(q.date<=hi)]
  if len(z):print(' ',lo,len(z),'IC %.6f ICIR %.6f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1)))
print('coverage',sig.notna().mean().mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
sig.to_csv('scripts/miner_2_20280425_lowvol_residual_signal.csv',index_label='date')
