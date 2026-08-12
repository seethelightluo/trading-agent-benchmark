import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=None
 try:d=get_index_daily_data(s,5000)
 except:pass
 if d is None or len(d)<300:
  try:d=get_stock_daily_data(s,5000)
  except:pass
 if d is not None and len(d):P[s]=d.set_index('date').close.astype(float)
px=pd.concat(P,axis=1).sort_index().ffill();r=px.pct_change();m=r.mean(axis=1)
f=pd.DataFrame(index=r.index,columns=r.columns,dtype=float)
for s in r.columns:
 # reward positive upside beta and penalize downside beta; covariance/variance conditional, lagged
 up=(m>0); dn=(m<0)
 bu=r[s].where(up).rolling(60,min_periods=20).cov(m.where(up))/m.where(up).rolling(60,min_periods=20).var()
 bd=r[s].where(dn).rolling(60,min_periods=20).cov(m.where(dn))/m.where(dn).rolling(60,min_periods=20).var()
 f[s]=(bu-bd).shift(1)
fr=px/px.shift(10)-1; rows=[]
for dt in f.index:
 x=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(x)>=8:rows.append((dt,x.iloc[:,0].corr(x.iloc[:,1]),len(x)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').dropna();q=a.ic
print('loaded',len(P),'dates',len(a),'avgN',a.n.mean(),'coverage',f.notna().sum().sum()/(f.shape[0]*len(U)))
print('IC %.8f ICIR %.8f hit %.4f turnover %.6f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),f.diff().abs().mean().mean()))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2025-12-31'),('2026-01-01','2028-12-31'),('2029-01-01','2030-12-31'),('2031-01-01','2032-12-31')]:
 z=q.loc[lo:hi];print(lo,hi,len(z),'IC %.8f ICIR %.8f'%(z.mean(),z.mean()/z.std(ddof=1)))
for n in [120,252,756]:
 z=q.tail(n);print('recent',n,'IC %.8f ICIR %.8f'%(z.mean(),z.mean()/z.std(ddof=1)))
f.stack().rename('signal').reset_index().to_csv('scripts/miner_2_20320527_beta_asym_signal.csv',index=False);a.reset_index().to_csv('scripts/miner_2_20320527_beta_asym_ic.csv',index=False)
