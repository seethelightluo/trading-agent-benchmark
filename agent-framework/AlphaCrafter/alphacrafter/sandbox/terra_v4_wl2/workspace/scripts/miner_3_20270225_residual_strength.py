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
# Relative-strength residual: asset 10d return minus contemporaneous cross-sectional median,
# normalized by its trailing 20d volatility; cross-sectional rank signal.
raw=r.rolling(10).sum(); resid=raw.sub(raw.median(axis=1),axis=0)
vol=r.rolling(20).std()*np.sqrt(20); sig=resid/vol
fwd={h:px.shift(-h)/px-1 for h in [1,5,10]}
def obs(s,h):
 vals=[]; ns=[]
 for dt in s.index:
  z=pd.concat([s.loc[dt],fwd[h].loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 x=np.asarray(vals); return len(x),np.mean(ns),np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0)
for h in [1,5,10]: print('H',h,'dates avgN IC ICIR hit',obs(sig,h))
print('coverage',sig.notna().sum().sum()/(len(U)*len(sig)),'calendar_dates',len(sig))
for lab,st,en in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027','2027','2027-12-31')]:
 q=sig.loc[(sig.index>=st)&(sig.index<=en)]; print(lab,'H5',obs(q,5))
