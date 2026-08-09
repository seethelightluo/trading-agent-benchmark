import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is not None and len(d):
  x=d.copy(); x.date=pd.to_datetime(x.date); D[s]=x.sort_values('date').set_index('date').close
P=pd.concat(D,axis=1).sort_index().ffill(); r=P.pct_change(); r3=P/P.shift(3)-1
# downside breadth: fraction of assets with negative 3d return, lagged extreme threshold
breadth=(r3<0).mean(axis=1); roll=breadth.shift(1).rolling(120,min_periods=60)
# extreme downside only (>= 70th percentile), use reversal on all assets, demean
cond=breadth.shift(1)>=roll.quantile(.70)
sig=-r3.where(cond, np.nan); sig=sig.sub(sig.median(axis=1),axis=0)
# align forward returns, daily and horizons
print('data',len(P), 'instruments',P.shape[1], 'date',P.index.min(),P.index.max())
for h in [1,5,10]:
 f=P.shift(-h)/P-1
 vals=[]
 for dt in sig.index:
  a=sig.loc[dt]; b=f.loc[dt]
  z=pd.concat([a,b],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('h',h,'dates',len(vals),'IC',np.nanmean(vals),'ICIR',np.nanmean(vals)/np.nanstd(vals,ddof=1),'hit',np.mean(np.array(vals)>0))
print('coverage',sig.notna().mean().mean(),'cond dates',cond.sum())
# turnover ranks on active dates
rank=sig.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean().mean())
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20261217_downside_breadth_reversal_signal.csv',index=False)
