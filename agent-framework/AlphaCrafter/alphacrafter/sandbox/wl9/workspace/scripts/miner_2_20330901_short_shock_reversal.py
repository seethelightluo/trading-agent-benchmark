import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cl={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  x=d[['date','close']].dropna().drop_duplicates('date').set_index('date')
  cl[s]=x.close.astype(float)
p=pd.DataFrame(cl).sort_index(); r=p.pct_change()
# Short-horizon shock reversal: recent 5D loss, normalized by trailing 20D risk;
# blend with a 20D range-location term to avoid treating ordinary noise as a shock.
vol=r.rolling(20,min_periods=12).std()*np.sqrt(252)
shock=-r.rolling(5,min_periods=4).sum()/(vol+0.05)
hi=p.rolling(60,min_periods=40).max(); lo=p.rolling(60,min_periods=40).min()
loc=((hi+lo)/2-p)/(hi-lo+1e-12)
f=(shock+0.35*loc).clip(-6,6).shift(1)
print('DATA dates',len(p),'instruments',len(cl),'range',p.index.min(),p.index.max())
for h in [10,20,40,60]:
 fr=p.shift(-h)/p-1; qs=[]; ns=[]; ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); ds.append(dt)
 q=pd.Series(qs,index=pd.to_datetime(ds)).dropna()
 print('H %d dates %d avgN %.2f IC %.6f ICIR %.6f hit %.4f'%(h,len(q),np.mean(ns),q.mean(),q.mean()/q.std(),(q>0).mean()))
 if h==60:
  for a,b in [('2020-01-01','2023-12-31'),('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2030-12-31'),('2031-01-01','2032-12-31'),('2033-01-01','2033-08-17')]:
   z=q.loc[a:b]
   if len(z): print('REGIME',a[:4],len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(),(z>0).mean()))
print('COVERAGE %.6f TURNOVER %.6f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20330901_short_shock_reversal_signal.csv',index=False)
