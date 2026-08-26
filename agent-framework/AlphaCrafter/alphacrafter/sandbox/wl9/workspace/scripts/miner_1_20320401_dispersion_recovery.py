import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4400)
 if d is not None and len(d)>300:
  d=d[['date','close']].dropna().drop_duplicates('date').set_index('date'); C[s]=d.close.astype(float)
p=pd.DataFrame(C).sort_index(); r=p.pct_change(); ret60=p/p.shift(60)-1
med=ret60.median(axis=1); loser=ret60.rsub(med,axis=0)
disp=ret60.sub(med,axis=0).abs().median(axis=1)
scale=(disp/disp.rolling(120,min_periods=60).median()).clip(0.5,2.0)
rec=(p/p.shift(10)-1)/(r.rolling(40,min_periods=25).std()*np.sqrt(10)+1e-8)
f=(loser.mul(1+0.25*rec.clip(-1,1),axis=0).mul(scale,axis=0)).shift(1)
for h in [5,10,20,40,60]:
 fr=p.shift(-h)/p-1; qs=[]; ns=[]; ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); ds.append(dt)
 q=pd.Series(qs,index=pd.to_datetime(ds)).dropna()
 print('H',h,'dates',len(q),'avgN %.2f'%np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(),(q>0).mean()))
 if h==60:
  for a,b in [('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2030-12-31'),('2031-01-01','2032-03-31')]:
   z=q.loc[a:b]; print('REGIME',a[:4],len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(),(z>0).mean()))
print('COVERAGE %.6f TURNOVER %.6f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20320401_dispersion_recovery_signal.csv',index=False)
