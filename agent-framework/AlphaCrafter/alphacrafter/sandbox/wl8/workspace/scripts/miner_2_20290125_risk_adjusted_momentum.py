import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
fs={}
for s in U:
 d=get_stock_daily_data(s,2600); x=d.copy(); x.date=pd.to_datetime(x.date); fs[s]=x.set_index('date').close.astype(float)
p=pd.concat(fs,axis=1).sort_index(); r=p.pct_change()
# medium-term risk-adjusted momentum: completed 60d return, normalized by lagged 20d realized volatility
sig=(p.pct_change(60)/r.rolling(20).std()).shift(1).clip(-8,8)
fr={h:p.shift(-h)/p-1 for h in [1,3,5,10]}
for h,y in fr.items():
 a=[]; ns=[]
 for dt in sig.index:
  q=sig.loc[dt]; z=y.loc[dt]; ok=q.notna()&z.notna()
  if ok.sum()>=8: a.append(q[ok].corr(z[ok])); ns.append(ok.sum())
 a=np.asarray(a); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1),6),'hit',round(np.mean(a>0),4))
print('coverage',round(sig.notna().sum().sum()/(len(sig)*15),4),'nonzero',round((sig.abs()>1e-12).sum().sum()/(len(sig)*15),4))
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20290125_risk_adjusted_momentum_signal.csv',index=False)
