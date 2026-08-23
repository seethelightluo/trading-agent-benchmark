import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
fs={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None: continue
 d=d.copy(); d.date=pd.to_datetime(d.date); fs[s]=d.set_index('date').close.astype(float)
p=pd.concat(fs,axis=1).sort_index(); r=p.pct_change()
# Reversal after a volatility expansion: completed 5d negative return, scaled by
# prior 20d risk and amplified when short volatility exceeds its 60d baseline.
vol20=r.rolling(20).std(); vol60=r.rolling(60).std()
sig=((-p.pct_change(5)/vol20)*(vol20/vol60).clip(.5,2.0)).shift(1).clip(-8,8)
print('assets',len(fs),'dates',len(p),'cutoff',p.index.max().date())
for h in [1,3,5,10]:
 y=p.shift(-h)/p-1; vals=[]; ns=[]
 for dt in sig.index:
  q,z=sig.loc[dt],y.loc[dt]; ok=q.notna()&z.notna()
  if ok.sum()>=8: vals.append(q[ok].corr(z[ok])); ns.append(int(ok.sum()))
 a=np.asarray(vals); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1),6),'hit',round(np.mean(a>0),4))
print('coverage',round(sig.notna().sum().sum()/(len(sig)*15),4))
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20290208_volatility_expansion_reversal_signal.csv',index=False)
