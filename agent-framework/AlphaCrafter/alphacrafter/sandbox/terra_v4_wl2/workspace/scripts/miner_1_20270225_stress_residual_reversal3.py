import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None:return x
  except Exception: pass
px=pd.DataFrame({s:get(s).set_index('date')['close'] for s in U}).sort_index(); r=px.pct_change()
# Residual reversal: remove cross-sectional common move, activate only after unusually high dispersion.
disp=r.median(axis=1); resid=r.sub(disp,axis=0)
rv=resid.rolling(3).sum(); d=r.std(axis=1); high=(d>d.rolling(90,min_periods=45).quantile(.75)).shift(1)
f=(-rv).where(high); f=f.sub(f.median(axis=1),axis=0)
fr={h:px.shift(-h)/px-1 for h in [1,3,5]}
allx={}
for h in [1,3,5]:
 a=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr[h].loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 a=np.array(a); allx[h]=a
 print('H',h,'dates',len(a),'avg_n',round(float(np.mean(ns)),2),'IC',round(float(np.mean(a)),6),'ICIR',round(float(np.mean(a)/np.std(a,ddof=1)),6),'hit',round(float(np.mean(a>0)),4))
for lab,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('online','2026-07-16','2027-02-25')]:
 q=[]
 for dt in f.index:
  if str(dt)>=lo and str(dt)<=hi:
   z=pd.concat([f.loc[dt],fr[1].loc[dt]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print(lab,'dates',len(q),'IC',round(float(np.mean(q)),6) if q else np.nan,'ICIR',round(float(np.mean(q)/np.std(q,ddof=1)),6) if len(q)>1 else np.nan)
print('coverage',float(f.notna().sum().sum()/f.size),'active_dates',int(high.sum()))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_1_20270225_stress_residual_reversal3.csv',index=False)
