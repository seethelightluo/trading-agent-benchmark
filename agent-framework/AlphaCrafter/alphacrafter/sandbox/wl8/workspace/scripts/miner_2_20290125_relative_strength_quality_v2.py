import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; fs={}
for s in U:
 d=get_stock_daily_data(s,2600); d.date=pd.to_datetime(d.date); fs[s]=d.set_index('date').close.astype(float)
p=pd.concat(fs,axis=1).sort_index(); r=p.pct_change(); market=r.mean(axis=1)
# asset 20d compounded return relative to equal-weight cross-asset 20d compounded return, inverse-vol scaled
asset20=(1+r).rolling(20).apply(np.prod,raw=True)-1
mkt20=(1+market).rolling(20).apply(np.prod,raw=True)-1
sig=((asset20-mkt20.to_frame().iloc[:,0] if hasattr(mkt20,'to_frame') else asset20-mkt20).div(r.rolling(20).std())).shift(1).clip(-8,8)
for h in [1,3,5,10]:
 y=p.shift(-h)/p-1; a=[]; ns=[]
 for dt in sig.index:
  q=sig.loc[dt]; z=y.loc[dt]; ok=q.notna()&z.notna()
  if ok.sum()>=8:a.append(q[ok].corr(z[ok]));ns.append(ok.sum())
 a=np.asarray(a); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1),6),'hit',round(np.mean(a>0),4))
print('coverage',round(sig.notna().sum().sum()/(len(sig)*15),4))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20290125_relative_strength_quality_v2_signal.csv',index=False)
