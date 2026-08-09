import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None:return x
  except Exception: pass
px=pd.DataFrame({s:g(s).set_index('date')['close'] for s in U}).sort_index(); r=px.pct_change(); fwd=px.shift(-5)/px-1
raw=r.rolling(5).sum(); vol=r.rolling(20).std(); f=-(raw/vol)
disp=r.std(axis=1); threshold=disp.rolling(60,min_periods=30).median(); f=f.where(disp>threshold)
a=[];ns=[];dates=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:
  a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));dates.append(dt)
a=np.array(a); print('horizon 5 dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2027-12-31')]:
 b=[]
 for dt in f.loc[lo:hi].index:
  z=pd.concat([f.loc[dt],(px.shift(-1)/px-1).loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:b.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 b=np.array(b); print('regime',lo,len(b),round(b.mean(),6) if len(b) else None,round(b.mean()/b.std(ddof=1),6) if len(b)>1 else None)
# Persist complete signal artifact for deterministic audit.
f.loc[dates].to_csv('../persistent/factor_signals_miner_2_20270225_dispersion_norm_reversal.csv',index_label='date')
print('artifact_rows',len(dates),'artifact_cols',len(U))
