import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None and len(x): return x
  except Exception: pass
px=pd.DataFrame({s:g(s).set_index('date')['close'] for s in U}).sort_index()
# Contrarian signal: negative cross-sectional rank of 60-session return ending 10 sessions ago.
raw=px.shift(10)/px.shift(70)-1
sig=-(raw.rank(axis=1,pct=True).sub(.5))
fwd=px.shift(-10)/px-1
vals=[];ns=[];rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:
  vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); rows.append([dt,*sig.loc[dt].reindex(U).values])
x=np.asarray(vals); print('dates',len(x),'avgN',np.mean(ns),'IC',np.mean(x),'ICIR',np.mean(x)/np.std(x,ddof=1),'hit',np.mean(x>0))
print('coverage',sig.notna().sum().sum()/(len(U)*len(sig)),'turnover',np.nanmean(np.abs(sig.diff()).sum(axis=1)/len(U)))
for lab,st,en in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027','2027','2027-12-31')]:
 q=[]
 for dt in sig.index:
  if str(dt)[:10]>=st and str(dt)[:10]<=en:
   z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=np.asarray(q);print(lab,len(q),np.mean(q) if len(q) else np.nan,np.mean(q)/np.std(q,ddof=1) if len(q)>1 else np.nan)
out=pd.DataFrame(rows,columns=['date']+U);out.to_csv('../persistent/factor_signals_miner_1_20270225_medium_momentum_60x10.csv',index=False)
