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
raw=r.rolling(10).sum(); resid=raw.sub(raw.median(axis=1),axis=0); vol=r.rolling(20).std()*np.sqrt(20)
sig=-(resid/vol) # contrarian residual strength, volatility normalized
fwd={h:px.shift(-h)/px-1 for h in [1,5,10]}
def obs(s,h):
 vals=[];ns=[]
 for dt in s.index:
  z=pd.concat([s.loc[dt],fwd[h].loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 x=np.asarray(vals);return len(x),float(np.mean(ns)),float(np.mean(x)),float(np.mean(x)/np.std(x,ddof=1)),float(np.mean(x>0))
for h in [1,5,10]: print('H',h,'dates avgN IC ICIR hit',obs(sig,h))
print('coverage',float(sig.notna().sum().sum()/(len(U)*len(sig))))
print('turnover',float(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_3_20270225_residual_reversal10.csv',index=False)
for lab,st,en in [('2025-26','2025','2026-12-31'),('2027','2027','2027-12-31')]:
 q=sig.loc[(sig.index>=st)&(sig.index<=en)];print(lab,'H5',obs(q,5))
