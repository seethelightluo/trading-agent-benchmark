import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None:return x
  except Exception: pass
px=pd.DataFrame({s:get(s).set_index('date')['close'] for s in U}).sort_index(); r=px.pct_change(); med=r.median(axis=1)
# market stress: prior 5d median return below zero, signal is reversal of prior 10d return
stress=(med.rolling(5).sum()<0).shift(1)
f=(-r.rolling(10).sum()).where(stress); f=f.sub(f.median(axis=1),axis=0)
for h in [1,3,5,10]:
 fr=px.shift(-h)/px-1; a=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(a); print('H',h,'dates',len(a),'avg_n',np.mean(ns) if ns else 0,'IC',np.mean(a) if len(a) else np.nan,'ICIR',np.mean(a)/np.std(a,ddof=1) if len(a)>1 else np.nan,'hit',np.mean(a>0) if len(a) else np.nan)
print('active_dates',int(stress.sum()),'total',len(stress))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_1_20270227_stress_reversal10.csv',index=False)
