import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,2600)
 if d is None or len(d)<180: d=get_index_daily_data(s,2600)
 if d is not None and len(d): rows.append(d[['date','close']].assign(symbol=s))
p=pd.concat(rows); w=p.pivot(index='date',columns='symbol',values='close').sort_index(); r=w.pct_change()
# Range-position trend quality: medium-term return is rewarded only when the asset
# is near the upper part of its trailing range, reducing noisy trend reversals.
ret=w.pct_change(40)
lo=w.rolling(120,min_periods=80).min(); hi=w.rolling(120,min_periods=80).max()
pos=((w-lo)/(hi-lo)).clip(0,1)
vol=r.rolling(20,min_periods=15).std()
f=(ret*(0.5+pos)/(vol*np.sqrt(252))).replace([np.inf,-np.inf],np.nan).clip(-10,10)
def calc(h,a=None,b=None):
 fut=w.shift(-h)/w-1; ds=f.index
 if a: ds=ds[(ds>=a)&(ds<=b)]
 q=[]; ns=[]
 for dt in ds:
  z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   x=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(x):q.append(x);ns.append(len(z))
 q=pd.Series(q)
 return len(q),q.mean(),q.std(ddof=1),q.mean()/q.std(ddof=1),(q>0).mean(),np.mean(ns)
print('cutoff',w.index.max().date(),'dates',len(w),'instruments',len(w.columns))
for h in [1,3,5,10]: print('H',h,'n meanIC std ICIR hit avgN',calc(h))
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-03-10')]: print('REG',a,b,calc(1,a,b))
print('coverage',f.notna().mean().mean(),'active_dates',f.notna().any(axis=1).sum())
rank=f.rank(axis=1,pct=True); print('rank_turnover',((rank-rank.shift()).abs().mean(axis=1)).mean())
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20270311_range_position_trend_signal.csv',index=False)
