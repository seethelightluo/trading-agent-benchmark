import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None:return x
  except Exception: pass
px=pd.DataFrame({s:g(s).set_index('date')['close'] for s in U}).sort_index(); r=np.log(px).diff(); fr={h:px.shift(-h)/px-1 for h in [1,5,10]}
d=g('DXY').set_index('date')['close'].reindex(px.index).ffill(); shock=(np.log(d).diff(5)>np.log(d).diff(5).rolling(252,min_periods=126).quantile(.75)).shift(1)
f=(-r.rolling(3).sum()).where(shock); f=f.sub(f.median(axis=1),axis=0)
s=f.stack().rename('signal').reset_index();s.columns=['date','symbol','signal'];s.to_csv('../persistent/factor_signals_miner_3_20270225_dxy_shock_reversal.csv',index=False)
print('active',int(shock.sum()),'coverage',f.notna().sum().sum()/(len(U)*len(f)))
for h in [1,5,10]:
 a=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr[h].loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.asarray(a);print(h,len(a),np.mean(ns) if ns else 0,np.mean(a) if len(a) else np.nan,np.mean(a)/np.std(a,ddof=1) if len(a)>1 else np.nan,np.mean(a>0) if len(a) else np.nan)
for label,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027','2027','2027-02-25')]:
 a=[]
 for dt in f.loc[lo:hi].index:
  z=pd.concat([f.loc[dt],fr[1].loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=np.asarray(a);print(label,len(a),np.mean(a) if len(a) else np.nan,np.mean(a)/np.std(a,ddof=1) if len(a)>1 else np.nan)
