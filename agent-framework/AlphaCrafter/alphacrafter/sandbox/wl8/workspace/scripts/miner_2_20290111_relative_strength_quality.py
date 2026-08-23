import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; fs={}
for s in U:
 d=get_stock_daily_data(s,2600); x=d.copy(); x.date=pd.to_datetime(x.date); fs[s]=x.set_index('date').close.astype(float)
p=pd.concat(fs,axis=1).sort_index(); r=p.pct_change(); m=r.mean(axis=1)
# residual relative strength: 20d asset return minus market return, quality scaled by inverse 20d vol
rel=(p.pct_change(20)-p.mean(axis=1).pct_change(20)).shift(1) # equivalent broad proxy
vol=r.rolling(20).std().shift(1); sig=(rel/vol).clip(-8,8)
for h in [1,3,5,10]:
 f=p.shift(-h)/p-1; a=[]; ns=[]
 for dt in sig.index:
  q=sig.loc[dt]; y=f.loc[dt]; ok=q.notna()&y.notna()
  if ok.sum()>=8:a.append(q[ok].corr(y[ok]));ns.append(ok.sum())
 a=np.array(a); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1),6),'hit',round(np.mean(a>0),4))
print('coverage',sig.notna().sum().sum()/(len(sig)*15)); out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20290111_relative_strength_quality_signal.csv',index=False)
