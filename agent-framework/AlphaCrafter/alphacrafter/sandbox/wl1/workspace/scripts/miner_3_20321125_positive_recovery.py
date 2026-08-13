import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-11-24'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None:
  d=d.copy();d.date=pd.to_datetime(d.date);raw[s]=d[d.date<=cut].set_index('date').sort_index().close
px=pd.DataFrame(raw).sort_index();r=np.log(px).diff();res=r.sub(r.mean(axis=1),axis=0)
loss=res.rolling(10,min_periods=8).sum(); rebound=res.rolling(3,min_periods=3).sum(); vol=res.rolling(30,min_periods=20).std().replace(0,np.nan); bench=r.mean(axis=1).rolling(20,min_periods=15).sum()
active=(bench>0).shift(1).fillna(False); f=(rebound.shift(1)-0.35*loss.shift(1)).div(vol.shift(1)).mul(active.astype(float),axis=0)
fr=np.log(px.shift(-10)/px);ics=[];ns=[];turn=[]
for i,d in enumerate(f.index):
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8:ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 if i:
  z=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turn.append(z.iloc[:,0].rank().sub(z.iloc[:,1].rank()).abs().mean()/len(z))
s=pd.Series(ics).dropna(); print('assets',len(raw),'valid',len(s),'avgN',np.mean(ns),'coverage',np.mean(np.array(ns)/len(U)),'IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',np.mean(s>0),'turn',np.mean(turn),'active',float((f.abs().sum(axis=1)>0).mean()))
for a,b in [('2024','2026'),('2027','2029'),('2030','2032'),('2032','2032')]:
 q=[]
 for d in f.index:
  if a<=str(d)[:4]<=b:
   z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
   if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(q).dropna(); print(a,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std())
out=f.where(f!=0).stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20321125_positive_recovery_signal.csv',index=False)
