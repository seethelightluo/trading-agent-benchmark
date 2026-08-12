import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<200: d=get_index_daily_data(s,4000)
 if d is not None and len(d): P[s]=d.set_index('date').close.astype(float)
pd_=pd.DataFrame(P).sort_index(); r=np.log(pd_).diff()
# Directional trend efficiency: signed cumulative return divided by path length; lagged one day.
ret=np.log(pd_/pd_.shift(30)); path=r.abs().rolling(30,min_periods=30).sum()
sig=(ret/path).replace([np.inf,-np.inf],np.nan).shift(1)
for h in [1,3,5,10]:
 a=[]; n=[]; f=np.log(pd_.shift(-h)/pd_)
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); n.append(len(z))
 x=pd.Series(a)
 print(f'h={h} dates={len(x)} avgN={np.mean(n):.2f} IC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1):.6f} hit={np.mean(x>0):.4f}')
print(f'coverage={sig.notna().sum().sum()/(len(sig)*len(U)):.4f} turnover={sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean():.6f} instruments={len(P)} rows={len(pd_)}')
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20291227_trend_efficiency30_signal.csv',index=False)
