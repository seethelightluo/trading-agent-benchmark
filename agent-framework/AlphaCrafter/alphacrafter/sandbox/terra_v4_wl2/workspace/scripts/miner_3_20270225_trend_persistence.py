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
# interpretable persistence: signed 20d trend times fraction of positive days, lagged one day
f=(r.rolling(20,min_periods=15).sum()*(r.gt(0).rolling(20,min_periods=15).mean()-0.5)).shift(1)
fr={h:px.shift(-h)/px-1 for h in [1,5,10]}
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_3_20270225_trend_persistence.csv',index=False)
print('dates',px.index.min(),px.index.max(),'assets',len(U))
for h in [1,5,10]:
 vals=[]; ns=[]; turns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr[h].loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(vals);print('H',h,'n_dates',len(a),'avg_n',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0),'coverage',len(a)/len(f))
for label,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027','2027','2027-02-25')]:
 a=[]
 for dt in f.loc[lo:hi].index:
  z=pd.concat([f.loc[dt],fr[1].loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=np.array(a);print(label,len(a),np.mean(a) if len(a) else np.nan,(np.mean(a)/np.std(a,ddof=1)) if len(a)>1 else np.nan)
# rank turnover
q=f.rank(axis=1,pct=True);print('turnover',np.nanmean((q-q.shift(1)).abs().mean(axis=1)))
