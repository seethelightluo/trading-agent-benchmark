import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,2300)
 if d is None or len(d)<150: d=get_index_daily_data(s,2300)
 if d is not None and len(d):
  x=d[['date','close']].copy(); x['symbol']=s; rows.append(x)
p=pd.concat(rows); wide=p.pivot(index='date',columns='symbol',values='close').sort_index(); r=wide.pct_change()
# Persistence-adjusted medium trend: 60d return, scaled by realized risk and by the fraction of positive daily moves.
vol=r.rolling(30,min_periods=20).std()*np.sqrt(252)
persistence=(r.gt(0).rolling(20,min_periods=15).mean()-0.5)*2
f=(wide.pct_change(60)/vol*persistence).replace([np.inf,-np.inf],np.nan)
def calc(h,start=None,end=None):
 fut=wide.shift(-h)/wide-1; qs=[]; ns=[]; idx=f.index
 if start: idx=idx[(idx>=start)&(idx<=end)]
 for dt in idx:
  z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   qs.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 q=pd.Series(qs).dropna(); return len(q),q.mean(),q.std(ddof=1),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean(),np.mean(ns)
print('cutoff',wide.index.max().date(),'dates',len(wide),'instruments',len(wide.columns))
for h in [1,3,5,10]: print('H',h,'n IC mean std ICIR hit avgN',calc(h))
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-12-31')]: print('REG',a,b,calc(1,a,b))
print('coverage',f.notna().mean().mean(),'active_dates',f.notna().any(axis=1).sum())
rank=f.rank(axis=1,pct=True); print('rank_turnover',((rank-rank.shift()).abs().mean(axis=1)).mean())
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20261231_persistence_trend_signal.csv',index=False)
