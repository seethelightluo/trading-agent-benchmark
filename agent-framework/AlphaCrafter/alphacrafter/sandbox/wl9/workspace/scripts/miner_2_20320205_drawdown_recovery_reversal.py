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
# Reversal signal: current drawdown from 60d high, normalized by the time since
# the high (capped), with a modest volatility normalization. Lagged one session.
high=p.rolling(60,min_periods=40).max(); dd=p/high-1
# recovery/drawdown depth per duration favors deep, recent pullbacks
age=pd.DataFrame(index=p.index,columns=p.columns,dtype=float)
for s in p:
 vals=high[s].to_numpy(); out=[]; last=-1
 for i,v in enumerate(vals):
  if np.isfinite(v) and (i==0 or not np.isfinite(vals[i-1]) or v>vals[i-1]+1e-12): last=i
  out.append(i-last+1 if last>=0 else np.nan)
 age[s]=out
vol=r.rolling(40,min_periods=20).std()*np.sqrt(40)
f=(-(dd)/(np.sqrt(age.clip(lower=1))+1e-8)/(vol+1e-8)).shift(1)
for h in [10,20,40,60]:
 fr=p.shift(-h)/p-1; qs=[]; ns=[]; ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); ds.append(dt)
 q=pd.Series(qs,index=pd.to_datetime(ds)).dropna()
 print('H',h,'dates',len(q),'avgN %.2f'%np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(),(q>0).mean()))
 if h==20:
  for a,b in [('2020-01-01','2023-12-31'),('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2032-01-31')]:
   z=q.loc[a:b]; print('REGIME',a[:4],len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(),(z>0).mean()))
print('COVERAGE %.6f TURNOVER %.6f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20320205_drawdown_recovery_reversal_signal.csv',index=False)
