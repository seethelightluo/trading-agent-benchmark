import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4400)
 if d is not None and len(d)>300:
  d=d[['date','close']].dropna().drop_duplicates('date').set_index('date'); C[s]=d.close.astype(float)
p=pd.DataFrame(C).sort_index(); r=p.pct_change(); vol=r.rolling(60,min_periods=40).std(); pos=r.gt(0).rolling(60,min_periods=40).mean()
f40=(-(pos-.5)/(vol+1e-8)).shift(1); pos20=r.gt(0).rolling(20,min_periods=14).mean(); f20=(-(pos20-.5)/(r.rolling(20,min_periods=14).std()+1e-8)).shift(1)
res=pd.DataFrame(index=f40.index,columns=f40.columns,dtype=float)
for dt in f40.index:
 z=pd.concat([f40.loc[dt],f20.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,1].var()>0:
  x=z.iloc[:,1].values; y=z.iloc[:,0].values; beta=np.cov(x,y,ddof=0)[0,1]/np.var(x); res.loc[dt,z.index]=y-beta*x
for h in [20,40,60]:
 fr=p.shift(-h)/p-1; qs=[]; ns=[]; ds=[]
 for dt in res.index:
  z=pd.concat([res.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); ds.append(dt)
 q=pd.Series(qs,index=pd.to_datetime(ds)).dropna(); print('H',h,'dates',len(q),'avgN %.2f'%np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(),(q>0).mean()))
 if h==40:
  for a,b in [('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2030-12-31'),('2031-01-01','2032-02-18')]:
   z=q.loc[a:b]; print('REGIME',a[:4],len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(),(z>0).mean()))
print('COVERAGE %.6f TURNOVER %.6f'%(res.notna().mean().mean(),res.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
out=res.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20320219_residual_inconsistency_40d_signal.csv',index=False)
