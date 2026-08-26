import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4500)
 if d is not None and len(d)>300: C[s]=d[['date','close']].dropna().drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(C).sort_index(); r=p.pct_change()
breadth=r.rolling(20,min_periods=12).mean().gt(0).mean(axis=1)
ret20=p.pct_change(20); prior20=p.shift(20).pct_change(20); vol60=r.rolling(60,min_periods=30).std()*np.sqrt(252)
raw=(ret20-prior20)/vol60; reg=(breadth-0.5)*2
sig=raw.mul(reg,axis=0).shift(1)
print('data',p.index.min(),p.index.max(),'rows',len(p),'instruments',len(C))
for h in [5,10,20,40,60]:
 fr=p.shift(-h)/p-1; ics=[]; ns=[]; ds=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(d)
 a=pd.Series(ics,index=pd.to_datetime(ds)).dropna()
 print(h,'dates',len(a),'avgN %.2f'%np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(),(a>0).mean()))
 if h==20:
  for aa,bb in [('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2030-12-31'),('2031-01-01','2032-05-12')]:
   z=a.loc[aa:bb]; print('REGIME',aa[:4],len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(),(z>0).mean()))
valid=sig.notna().sum(axis=1); print('coverage %.6f turnover %.6f valid_dates %d'%(sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),(valid>=8).sum()))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20320513_breadth_acceleration_signal.csv',index=False)
