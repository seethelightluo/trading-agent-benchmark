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
# Persistence factor: trailing signed return multiplied by breadth of positive daily returns,
# shifted one completed day to avoid look-ahead.
f=(r.rolling(20,min_periods=15).sum()*(r.gt(0).rolling(20,min_periods=15).mean()-0.5)).shift(1)
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('../persistent/factor_signals_miner_1_20270225_trend_persistence.csv',index=False)
print('dates',px.index.min(),px.index.max(),'assets',px.notna().sum(axis=1).mean(),'rows',len(out))
for h in [1,5,10]:
 fw=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 a=pd.Series(vals).dropna(); print('H',h,'n_dates',len(a),'avg_n',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
for label,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027','2027','2027-02-25')]:
 fw=px.shift(-1)/px-1; a=[]
 for dt in f.loc[lo:hi].index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=pd.Series(a).dropna();print(label,'n',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1))
q=f.rank(axis=1,pct=True);print('turnover',np.nanmean((q-q.shift(1)).abs().mean(axis=1)),'coverage',len(vals)/len(f))
