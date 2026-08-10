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
# Market-wide breadth stress, lagged one day; cross-sectional reversal signal
breadth=(r>0).mean(axis=1); stress=(breadth<=breadth.rolling(60,min_periods=30).quantile(.25)).shift(1)
f=(-r.rolling(5).sum()).where(stress); f=f.sub(f.median(axis=1),axis=0)
fr=px.shift(-1)/px-1
ics=[]; ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
a=np.array(ics)
print('dates',len(a),'avg_n',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0),'active',int(stress.sum()),'coverage',f.notna().sum().sum()/f.size)
for lab,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027','2027','2027-02-25')]:
 q=[x for d,x in zip(f.index,[]) ]
 # recompute paired dates
 vals=[]
 for dt in f.index:
  if str(dt)<lo or str(dt)>hi: continue
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print(lab,len(vals),np.mean(vals) if vals else np.nan,np.mean(vals)/np.std(vals,ddof=1) if len(vals)>1 else np.nan)
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_2_20270225_breadth_stress_reversal.csv',index=False)
