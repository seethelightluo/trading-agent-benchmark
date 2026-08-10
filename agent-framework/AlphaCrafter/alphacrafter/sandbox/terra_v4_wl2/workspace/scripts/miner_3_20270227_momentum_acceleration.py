import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None and len(x): return x
  except Exception: pass
px=pd.DataFrame({s:g(s).set_index('date')['close'] for s in U}).sort_index(); r=px.pct_change()
# Acceleration: recent 5d return relative to preceding 15d return, cross-section demeaned.
a=r.rolling(5).sum()-r.shift(5).rolling(15).sum(); sig=a.sub(a.median(axis=1),axis=0)
fwd={h:px.shift(-h)/px-1 for h in [1,5,10]}
for h in [1,5,10]:
 vals=[];ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd[h].loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 x=np.asarray(vals);print('H',h,'dates',len(x),'avgN',np.mean(ns),'IC',np.mean(x),'ICIR',np.mean(x)/np.std(x,ddof=1),'hit',np.mean(x>0))
print('coverage',sig.notna().sum().sum()/(len(U)*len(sig)),'total',len(sig))
for lab,st,en in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027','2027','2027-12-31')]:
 x=[]
 for dt in sig.index:
  if str(dt)[:10]>=st and str(dt)[:10]<=en:
   z=pd.concat([sig.loc[dt],fwd[1].loc[dt]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1:x.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 x=np.asarray(x);print(lab,len(x),np.mean(x) if len(x) else np.nan,np.mean(x)/np.std(x,ddof=1) if len(x)>1 else np.nan)
