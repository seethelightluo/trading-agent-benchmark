import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None:return x
  except: pass
px=pd.DataFrame({s:get(s).set_index('date').close for s in U}).sort_index(); r=px.pct_change()
# A slower, persistent stress breadth regime, lagged one day to avoid look-ahead.
b=((r.rolling(3).sum()<0).sum(axis=1)/r.rolling(3).sum().notna().sum(axis=1)).shift(1)
for th in [.60,.67,.75,.80]:
 active=b>=th; f=(-r.rolling(3).sum()).where(active); f=f.sub(f.median(axis=1),axis=0)
 fr=px.shift(-1)/px-1; vals=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.asarray(vals); print('threshold',th,'dates',len(a),'avgN',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0),'active',active.sum())
 if th==.75:
  out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_3_20270225_persistent_stress_reversal.csv',index=False)
