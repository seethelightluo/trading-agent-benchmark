import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<200: d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  x=d[['date','close']].drop_duplicates('date'); x['symbol']=s; rows.append(x)
cl=pd.concat(rows).pivot(index='date',columns='symbol',values='close').sort_index().ffill()
r=cl.pct_change(); vol=r.rolling(20,min_periods=15).std()*np.sqrt(252)
# intermediate trend signal: 10-session return risk-adjusted, lagged one completed day
f=(cl.pct_change(10)/vol.replace(0,np.nan)).shift(1)
fut=cl.shift(-1)/cl-1
qs=[]; ns=[]
for dt in cl.index:
 z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q): qs.append(q); ns.append(len(z))
q=pd.Series(qs)
print('cutoff',cl.index.max().date(),'dates',len(cl),'instruments',len(cl.columns),'obs',len(q),'avgN',np.mean(ns),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(len(q)),'hit',(q>0).mean(),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20280210_rap10_signal.csv',index=False)
