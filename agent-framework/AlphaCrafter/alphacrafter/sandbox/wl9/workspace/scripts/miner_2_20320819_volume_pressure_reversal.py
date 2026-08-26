import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cl={}; mf={}
for s in U:
 d=get_stock_daily_data(s,days=4600)
 if d is None or len(d)<300: continue
 d=d[['date','close','high','low','volume']].dropna().drop_duplicates('date').set_index('date')
 cl[s]=d.close.astype(float)
 rng=(d.high-d.low).replace(0,np.nan)
 # Dollar-volume weighted close-location pressure, robust to noisy single sessions.
 loc=((2*d.close-d.high-d.low)/rng).clip(-1,1)
 mf[s]=(loc*d.volume).rolling(20,min_periods=15).sum()/d.volume.rolling(20,min_periods=15).sum()
p=pd.DataFrame(cl).sort_index(); m=pd.DataFrame(mf).reindex(p.index); r=p.pct_change()
vol=r.rolling(60,min_periods=40).std()*np.sqrt(252)
f=(-m/vol.replace(0,np.nan)).shift(1)
print('DATA dates',len(p),'instruments',len(cl),'range',p.index.min(),p.index.max())
for h in [5,10,20,40,60]:
 fr=p.shift(-h)/p-1; qs=[]; ns=[]; ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); ds.append(dt)
 q=pd.Series(qs,index=pd.to_datetime(ds)).dropna()
 print('H',h,'dates',len(q),'avgN %.2f'%np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(),(q>0).mean()))
 if h==60:
  for a,b in [('2020-01-01','2023-12-31'),('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2030-12-31'),('2031-01-01','2032-08-18')]:
   z=q.loc[a:b]
   if len(z): print('REGIME',a[:4],len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(),(z>0).mean()))
print('COVERAGE %.6f TURNOVER %.6f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20320819_volume_pressure_reversal_signal.csv',index=False)
