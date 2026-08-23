import os, numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for f in (get_stock_daily_data,get_index_daily_data):
  try:
   z=f(s,5000)
   if z is not None and len(z)>100:return z
  except:pass
D={s:get(s) for s in U};D={s:z for s,z in D.items() if z is not None}
v=pd.read_csv('../persistent/index_data/VIX.csv');v['date']=pd.to_datetime(v['date']);v=v.set_index('date')['close'].astype(float)
C=pd.DataFrame({s:x.set_index(pd.to_datetime(x.date)).close for s,x in D.items()}); ret=C.pct_change(); vr=v.pct_change()
# contrarian asset exposure to VIX shocks: assets with lower rolling beta are defensive; signal favors negative beta
beta=ret.rolling(60).cov(vr).div(vr.rolling(60).var(),axis=0)
# lag and gate on elevated VIX level: defensive beta signal only when VIX > its 120d median
sig=(-beta).where(v.reindex(C.index).rolling(120).mean().gt(v.reindex(C.index).rolling(120).median()),0).shift(1)
R=(C.shift(-10)/C-1).reindex(sig.index)
ics=[]; ns=[]
for d in sig.index:
 z=pd.concat([sig.loc[d],R.loc[d]],axis=1).dropna()
 if len(z)>=8:ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
ic=pd.Series(ics).dropna()
print('dates',len(ic),'avgN',np.mean(ns),'coverage',sig.notna().sum().sum()/(sig.shape[0]*len(U)))
print('IC',ic.mean(),'ICIR',ic.mean()/ic.std(),'hit',(ic>0).mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
for n in [180,365]:
 q=ic.tail(n);print('recent',n,q.mean(),q.mean()/q.std() if q.std() else np.nan)
for h in [1,5,10,20]:
 rr=(C.shift(-h)/C-1).reindex(sig.index);a=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],rr.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(a),len(a))
os.makedirs('scripts',exist_ok=True);sig.to_csv('scripts/miner_1_20311002_vix_beta_defensive_signal.csv');ic.rename('ic').to_csv('scripts/miner_1_20311002_vix_beta_defensive_ic.csv')
