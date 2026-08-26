import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,days=4500)
 if d is not None and len(d)>300: D[s]=d[['date','high','low','close']].dropna().drop_duplicates('date').set_index('date').astype(float)
idx=sorted(set().union(*[set(x.index) for x in D.values()])); h=pd.DataFrame({s:x.high for s,x in D.items()},index=idx); l=pd.DataFrame({s:x.low for s,x in D.items()},index=idx); c=pd.DataFrame({s:x.close for s,x in D.items()},index=idx)
clv=((2*c-h-l)/(h-l).replace(0,np.nan)).clip(-1,1); pressure=clv.ewm(span=20,min_periods=15).mean(); body=((c-c.shift(1))/(h-l).replace(0,np.nan)).rolling(10,min_periods=7).mean()
f=-(0.7*pressure+0.3*body).shift(1)
print('DATA dates',len(c),'instruments',len(D),'range',c.index.min(),c.index.max())
for horizon in [5,10,20,40,60]:
 fr=c.shift(-horizon)/c-1; qs=[]; ns=[]; dates=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); dates.append(dt)
 q=pd.Series(qs,index=pd.to_datetime(dates)).dropna(); print('H',horizon,'dates',len(q),'avgN %.2f'%np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(),(q>0).mean()))
 if horizon==60:
  for a,b in [('2020-01-01','2023-12-31'),('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2030-12-31'),('2031-01-01','2032-07-07')]:
   z=q.loc[a:b]
   if len(z): print('REGIME',a[:4],len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(),(z>0).mean()))
print('COVERAGE %.6f TURNOVER %.6f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20320708_intraday_pressure_reversal_signal.csv',index=False)
