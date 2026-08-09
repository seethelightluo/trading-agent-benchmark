import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); D[s]=d.sort_values('date').set_index('date').close
P=pd.concat(D,axis=1).sort_index().ffill(); r=P.pct_change(); r10=P/P.shift(10)-1
breadth=(r10>0).mean(axis=1)
q=breadth.shift(1).rolling(120,min_periods=60).quantile(.70)
cond=breadth.shift(1)>=q
sig=r10.where(cond,np.nan); sig=sig.sub(sig.median(axis=1),axis=0)
print('data',len(P),'instruments',P.shape[1],'period',P.index.min(),P.index.max(),'condition_dates',int(cond.sum()))
for h in [1,5,10]:
 f=P.shift(-h)/P-1; vals=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
 vals=np.asarray(vals,float)
 print('h',h,'dates',len(vals),'avg_n',P.shape[1],'IC',np.nanmean(vals),'ICIR',np.nanmean(vals)/np.nanstd(vals,ddof=1),'hit',np.mean(vals>0))
print('coverage',sig.notna().mean().mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20261217_upside_breadth_momentum_signal.csv',index=False)
