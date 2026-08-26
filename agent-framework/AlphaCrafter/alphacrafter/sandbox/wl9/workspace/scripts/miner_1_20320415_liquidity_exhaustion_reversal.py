import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={};V={}
for s in U:
 d=get_stock_daily_data(s,days=4400)
 if d is not None and len(d)>300:
  d=d[['date','close','volume']].dropna(subset=['close']).drop_duplicates('date').set_index('date')
  P[s]=d.close.astype(float); V[s]=d.volume.astype(float).replace(0,np.nan)
p=pd.DataFrame(P).sort_index(); v=pd.DataFrame(V).reindex(p.index); r=p.pct_change()
# Liquidity-exhaustion reversal: recent underperformance, strengthened by abnormal
# volume and by a one-session shock, all lagged one completed session.
ret10=p/p.shift(10)-1
ret3=p/p.shift(3)-1
rv=r.rolling(40,min_periods=25).std()
vr=(v.rolling(10,min_periods=7).mean()/(v.rolling(60,min_periods=35).mean()+1e-12)-1).clip(-.75,2.0)
shock=(-ret3/(rv*np.sqrt(3)+1e-8)).clip(-3,3)
f=((-ret10/(rv+1e-8))*(1+0.35*vr.clip(lower=0))*(1+0.15*shock.clip(lower=0))).shift(1)
for h in [5,10,20,40,60]:
 fr=p.shift(-h)/p-1; qs=[]; ns=[]; ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); ds.append(dt)
 q=pd.Series(qs,index=pd.to_datetime(ds)).dropna()
 print('H',h,'dates',len(q),'avgN %.2f'%np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(),(q>0).mean()))
 if h==20:
  for a,b in [('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2030-12-31'),('2031-01-01','2032-03-31')]:
   z=q.loc[a:b]; print('REGIME',a[:4],len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(),(z>0).mean()))
print('COVERAGE %.6f TURNOVER %.6f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20320415_liquidity_exhaustion_reversal_signal.csv',index=False)
