import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4500)
 if d is not None and len(d)>300: C[s]=d[['date','close']].dropna().drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(C).sort_index(); r=p.pct_change()
# Novel interpretable signal: fade medium-term relative losers, but only when recovery
# persists across two nested horizons; scale by recent volatility. Shift one day.
rel=r.rolling(30,min_periods=20).sum(); rel=rel.sub(rel.median(axis=1),axis=0)
rec10=r.rolling(10,min_periods=8).sum(); rec20=r.rolling(20,min_periods=15).sum()
vol=r.rolling(40,min_periods=25).std()
# positive recovery persistence, with stronger weight for assets whose 20d recovery is positive
recovery=(rec10.clip(lower=0)/(.03+vol*np.sqrt(10))).clip(0,2)
persist=(rec20>0).astype(float)
f=(-rel*(1+0.65*recovery)*(0.75+0.5*persist)/(vol*np.sqrt(40)+0.01)).shift(1)
for h in [5,10,20,40,60]:
 fr=p.shift(-h)/p-1; qs=[]; ns=[]; dates=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); dates.append(dt)
 q=pd.Series(qs,index=pd.to_datetime(dates)).dropna()
 print('H',h,'dates',len(q),'avgN %.2f'%np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(),(q>0).mean()))
 if h==60:
  for a,b in [('2020-01-01','2023-12-31'),('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2030-12-31'),('2031-01-01','2032-06-09')]:
   z=q.loc[a:b]
   if len(z): print('REGIME',a[:4],len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(),(z>0).mean()))
print('COVERAGE %.6f TURNOVER %.6f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20320610_recovery_persistence_reversal_signal.csv',index=False)
