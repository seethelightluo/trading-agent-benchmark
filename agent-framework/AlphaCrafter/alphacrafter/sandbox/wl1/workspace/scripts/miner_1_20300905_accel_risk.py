import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None: d=get_index_daily_data(s,4000)
 if d is not None: px[s]=d.sort_values('date').set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill(); R=P.pct_change()
# Acceleration: recent 20d trend versus its prior 40d pace, scaled by recent realized risk.
r20=P.pct_change(20); r60=P.pct_change(60); vol=R.rolling(40,min_periods=20).std()*np.sqrt(252)
raw=(r20-r60/3)/(vol+1e-8)
sig=raw.rank(axis=1,pct=True).shift(1)
for h in [1,5,10,20]:
 a=[]; ns=[]
 for dt in sig.index:
  y=P.shift(-h).loc[dt]/P.loc[dt]-1; z=pd.concat([sig.loc[dt],y],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 a=np.array(a); print(f'{h}d dates={len(a)} avgN={np.mean(ns):.2f} IC={np.nanmean(a):.6f} ICIR={np.nanmean(a)/np.nanstd(a,ddof=1):.6f} hit={np.mean(a>0):.4f}')
print('coverage %.4f turnover %.6f period %s %s'%(np.isfinite(sig).mean().mean(),sig.diff().abs().mean(axis=1).dropna().mean(),P.index.min(),P.index.max()))
sig.index.name='date'; sig.to_csv('scripts/miner_1_20300905_accel_risk_signal.csv')
