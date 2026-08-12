import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-05-26'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); d=d[d.date<=cut].set_index('date').sort_index(); raw[s]=d.close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); bench=r.mean(axis=1); resid=r.sub(bench,axis=0)
rv=resid.rolling(20,min_periods=15).std(); pull=(-resid.rolling(10,min_periods=8).sum()/rv.rolling(60,min_periods=30).mean()).shift(1)
disp=resid.std(axis=1).rolling(20,min_periods=15).mean(); ratio=disp/disp.rolling(180,min_periods=90).median(); f=pull.mul(ratio.clip(.6,1.8),axis=0)
fr=np.log(px.shift(-10)/px); vals=[]; ns=[]; turns=[]
for i,dt in enumerate(f.index):
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 if i:
  q=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
  if len(q)>=8: turns.append(q.iloc[:,0].rank().sub(q.iloc[:,1].rank()).abs().mean()/len(q))
s=pd.Series(vals); print('assets',len(raw),'dates',len(px),'valid',len(s),'avgN',np.mean(ns),'IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',np.mean(s>0),'turn',np.mean(turns),'coverage',np.mean(f.notna().sum(axis=1)/15))
for a,b in [('2024','2026'),('2027','2029'),('2030','2032')]:
 v=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if a<=str(d)[:4]<=b and len(z)>=8:v.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(v); print(a,b,len(q),q.mean(),q.mean()/q.std())
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20320527_smooth_pullback_signal.csv',index=False)
