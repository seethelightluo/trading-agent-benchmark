import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4400)
 if d is not None and len(d)>300:
  d=d[['date','close']].dropna().drop_duplicates('date').set_index('date')
  C[s]=d.close.astype(float)
p=pd.DataFrame(C).sort_index(); r=p.pct_change()
# Recovery-conviction reversal: contrarian 20d loss, scaled by distance below 60d high
# and by the fraction of recent up days, which favors assets beginning to recover.
ret20=p/p.shift(20)-1
vol=r.rolling(60,min_periods=40).std()
dd=p/p.rolling(60,min_periods=40).max()-1
up=r.gt(0).rolling(20,min_periods=14).mean()
# positive signal = expected rebound; lag all observable components one session
f=((-ret20/(vol+1e-8))*(1+(-dd).clip(0,0.35))* (0.5+up)).shift(1)
for h in [5,10,20,40,60]:
 fr=p.shift(-h)/p-1; qs=[]; ns=[]; ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(dt)
 q=pd.Series(qs,index=pd.to_datetime(ds)).dropna()
 print('H',h,'dates',len(q),'avgN %.2f'%np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(),(q>0).mean()))
 if h==20:
  for a,b in [('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2030-12-31'),('2031-01-01','2032-03-17')]:
   z=q.loc[a:b]; print('REGIME',a[:4],len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(),(z>0).mean()))
  print('COVERAGE %.6f TURNOVER %.6f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20320318_recovery_conviction_signal.csv',index=False)
