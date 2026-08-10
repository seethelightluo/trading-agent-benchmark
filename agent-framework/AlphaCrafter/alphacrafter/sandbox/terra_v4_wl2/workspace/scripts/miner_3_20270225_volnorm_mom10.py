import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   x=f(s,days=5000)
   if x is not None and len(x): return x
  except Exception: pass
px=pd.DataFrame({s:get(s).set_index('date')['close'] for s in U}).sort_index()
r=px.pct_change(); fwd=px.shift(-1)/px-1
# Novel candidate: volatility-normalized intermediate momentum, using only lagged observations.
vol=r.rolling(20,min_periods=15).std(); sig=r.rolling(10,min_periods=10).sum()/vol
# winsorize cross-section to limit crypto/commodity scale effects
sig=sig.clip(lower=sig.quantile(.1,axis=1),upper=sig.quantile(.9,axis=1),axis=0).shift(1)
for h in [1,5,10]:
 y=px.shift(-h)/px-1; vals=[]; ns=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 a=np.array(vals); print('horizon',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1),6),'hit',round(np.mean(a>0),4),'coverage',round(sig.notna().sum().sum()/(len(U)*len(sig)),4))
for lo,hi in [(pd.Timestamp('2020-01-01'),pd.Timestamp('2022-12-31')),(pd.Timestamp('2023-01-01'),pd.Timestamp('2024-12-31')),(pd.Timestamp('2025-01-01'),pd.Timestamp('2026-12-31')),(pd.Timestamp('2027-01-01'),pd.Timestamp('2027-02-25'))]:
 vals=[]
 for d in sig.loc[lo:hi].index:
  z=pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('regime',str(lo.date()),str(hi.date()),'n',len(vals),'IC',round(np.mean(vals),6) if vals else None)
print('artifact rows',len(sig),'cols',len(U))
# save recoverable signal artifact
out=sig.reset_index().rename(columns={'date':'timestamp'}); out.to_csv('../persistent/factor_signals_miner_3_20270225_volnorm_mom10.csv',index=False)
