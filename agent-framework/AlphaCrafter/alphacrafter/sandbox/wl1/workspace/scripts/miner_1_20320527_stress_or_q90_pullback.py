import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-05-26'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); d=d[d.date<=cut].set_index('date').sort_index(); raw[s]=d.close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); bench=r.mean(axis=1); resid=r.sub(bench,axis=0)
vol=r.rolling(20,min_periods=15).std(); base=(-resid.rolling(10,min_periods=8).sum()/vol.rolling(20,min_periods=15).mean()).shift(1)
rd=resid.std(axis=1); q=rd.rolling(120,min_periods=60).quantile(.90); stress=(bench.rolling(20,min_periods=15).sum()<0)|(rd>q)
f=base.where(stress.shift(1)); fr=np.log(px.shift(-10)/px)
vals=[];ns=[];turns=[];dates=[]
for i,dt in enumerate(f.index):
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));dates.append(dt)
 if i:
  qx=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
  if len(qx)>=8: turns.append(qx.iloc[:,0].rank().sub(qx.iloc[:,1].rank()).abs().mean()/len(qx))
s=pd.Series(vals,index=dates);print('assets',len(raw),'calendar_dates',len(px),'valid_dates',len(s),'avg_instruments',np.mean(ns),'coverage',np.mean(ns)/len(U));print('IC10 %.6f ICIR %.6f hit %.4f turnover %.4f'%(s.mean(),s.mean()/s.std(),np.mean(s>0),np.mean(turns)));print('active_frac',f.notna().any(axis=1).mean())
for a,b in [('2024','2026'),('2027','2029'),('2030','2032')]:
 qx=s[(s.index.year>=int(a))&(s.index.year<=int(b))];print(a,b,'dates',len(qx),'IC',qx.mean(),'ICIR',qx.mean()/qx.std())
for h in [1,5,20]:
 rr=np.log(px.shift(-h)/px); vv=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],rr.loc[dt]],axis=1).dropna()
  if len(z)>=8:vv.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('horizon',h,'IC',np.mean(vv),'dates',len(vv))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20320527_stress_or_q90_pullback_signal.csv',index=False)
