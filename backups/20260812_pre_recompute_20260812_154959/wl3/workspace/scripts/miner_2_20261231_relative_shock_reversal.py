import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,2300)
 if d is None or len(d)<100: d=get_index_daily_data(s,2300)
 if d is not None and len(d): rows.append(d[['date','close']].assign(symbol=s))
p=pd.concat(rows); w=p.pivot(index='date',columns='symbol',values='close').sort_index(); r=w.pct_change()
# Relative shock reversal: reverse each asset's 3-day return after removing the same-day
# cross-sectional move, scaled by its trailing 20-day realized volatility.
raw=r.rolling(3,min_periods=3).sum(); rel=raw.sub(raw.median(axis=1),axis=0)
vol=r.rolling(20,min_periods=12).std()*np.sqrt(20)
f=(-rel/vol).replace([np.inf,-np.inf],np.nan)
def calc(h,a=None,b=None):
 fut=w.shift(-h)/w-1; dates=f.index
 if a: dates=dates[(dates>=a)&(dates<=b)]
 qs=[]; ns=[]
 for dt in dates:
  z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): qs.append(q); ns.append(len(z))
 q=pd.Series(qs); return len(q),q.mean(),q.std(ddof=1),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean(),np.mean(ns)
print('cutoff',w.index.max().date(),'dates',len(w),'instruments',len(w.columns))
for h in [1,3,5,10]: print('H',h,'n meanIC std ICIR hit avgN',calc(h))
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-12-31')]: print('REG',a,b,calc(1,a,b))
print('coverage',f.notna().mean().mean(),'active_dates',f.notna().any(axis=1).sum())
rank=f.rank(axis=1,pct=True); print('rank_turnover',((rank-rank.shift()).abs().mean(axis=1)).mean())
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20261231_relative_shock_reversal_signal.csv',index=False)
