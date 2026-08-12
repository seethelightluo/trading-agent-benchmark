import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-04-14'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); d=d[d.date<=cut].sort_values('date'); raw[s]=d.set_index('date').close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); bench=r.mean(axis=1); resid=r.sub(bench,axis=0)
# Stress-conditioned residual pullback: contrarian 10d residual return, volatility normalized,
# activated only when the equal-weight benchmark has a negative 20d return.
base=(-resid.rolling(10,min_periods=8).sum()/r.rolling(20,min_periods=15).std().rolling(20,min_periods=15).mean()).shift(1)
stress=(bench.rolling(20,min_periods=15).sum()<0).shift(1)
f=base.where(stress)
fr={h:np.log(px.shift(-h)/px) for h in [1,5,10,20]}
def calc(h):
 vals=[]; ns=[]; turns=[]
 for i,dt in enumerate(f.index):
  z=pd.concat([f.loc[dt],fr[h].loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
  if i:
   q=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
   if len(q)>=8: turns.append(q.iloc[:,0].rank().sub(q.iloc[:,1].rank()).abs().mean()/len(q))
 s=pd.Series(vals); return len(s),np.mean(ns),s.mean(),s.mean()/s.std(),np.mean(s>0),np.mean(turns)
print('assets',len(raw),'dates',len(px),'active',float(stress.mean()))
for h in [1,5,10,20]: print('h',h,calc(h))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032')]:
 v=[]
 for dt in f.index:
  if a<=str(dt)[:4]<=b:
   z=pd.concat([f.loc[dt],fr[10].loc[dt]],axis=1).dropna()
   if len(z)>=8:v.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 s=pd.Series(v); print('regime',a,b,len(s),round(s.mean(),5),round(s.mean()/s.std(),5),round(np.mean(s>0),5))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20320415_stress_pullback_signal.csv',index=False)
