import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-08-04'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); d=d[d.date<=cut].set_index('date').sort_index(); raw[s]=d.close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); bench=r.mean(axis=1); resid=r.sub(bench,axis=0)
# Rebound-after-stress: lagged 20d residual losers with a fresh 3d rebound, active in weak benchmark regimes.
r20=resid.rolling(20,min_periods=15).sum(); r3=resid.rolling(3,min_periods=3).sum()
rv=resid.rolling(20,min_periods=15).std(); stress=(bench.rolling(20,min_periods=15).sum()<0) | (resid.std(axis=1).rolling(60,min_periods=30).mean()>resid.std(axis=1).rolling(120,min_periods=60).median())
f=((-r20/(rv*np.sqrt(20))) + 0.5*(r3/(rv*np.sqrt(3)))).shift(1).where(stress.shift(1).fillna(False),np.nan)
fr=np.log(px.shift(-10)/px); ics=[]; ns=[]; turns=[]
for i,dt in enumerate(f.index):
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 if i:
  z=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
  if len(z)>=8: turns.append(z.iloc[:,0].rank().sub(z.iloc[:,1].rank()).abs().mean()/len(z))
s=pd.Series(ics); print('assets',len(raw),'calendar_dates',len(px),'valid_dates',len(s),'avgN',np.mean(ns),'active_frac',float(f.notna().any(axis=1).mean())); print('IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',np.mean(s>0),'turn',np.mean(turns))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032')]:
 v=[]
 for d in f.index:
  if a<=str(d)[:4]<=b:
   z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
   if len(z)>=8:v.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(v); print(a,b,'n',len(q),'IC',q.mean() if len(q) else np.nan,'ICIR',q.mean()/q.std() if len(q)>1 else np.nan)
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20320805_rebound_stress_signal.csv',index=False)
