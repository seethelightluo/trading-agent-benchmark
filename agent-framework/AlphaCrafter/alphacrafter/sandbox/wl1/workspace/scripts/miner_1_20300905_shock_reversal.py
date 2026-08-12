import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None: d=get_index_daily_data(s,4000)
 if d is not None: px[s]=d.sort_values('date').set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill(); R=P.pct_change(); v=R.rolling(20,min_periods=10).std()
# Volatility-conditioned short-term reversal: emphasize unusually large prior shocks,
# while retaining a small 5-day reversal component. Everything is lagged one session.
z=(R.shift(1)/(v.shift(1)+1e-8)).abs()
raw=(-R.shift(1)/(v.shift(1)+1e-8))*(1+0.75*(z>1.25).astype(float)) + 0.15*(-P.pct_change(5).shift(1)/(v.shift(1)+1e-8))
sig=raw.rank(axis=1,pct=True).shift(1)
for h in [1,5,10,20]:
 a=[]; ns=[]
 for dt in sig.index:
  y=P.shift(-h).loc[dt]/P.loc[dt]-1; q=pd.concat([sig.loc[dt],y],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   a.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman')); ns.append(len(q))
 a=np.array(a); print(f'{h}d dates={len(a)} avgN={np.mean(ns):.2f} IC={np.nanmean(a):.6f} ICIR={np.nanmean(a)/np.nanstd(a,ddof=1):.6f} hit={np.mean(a>0):.4f}')
print('coverage %.4f turnover %.6f period %s %s'%(np.isfinite(sig).mean().mean(),sig.diff().abs().mean(axis=1).dropna().mean(),P.index.min(),P.index.max()))
sig.index.name='date'; sig.to_csv('scripts/miner_1_20300905_shock_reversal_signal.csv')
