import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<200: d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  x=d[['date','close']].drop_duplicates('date'); x['symbol']=s; rows.append(x)
cl=pd.concat(rows).pivot(index='date',columns='symbol',values='close').sort_index().ffill(); r=cl.pct_change()
# Volatility-scaled short reversal: negative 5-session return divided by trailing 20-session realized volatility, lagged one day.
f=(-(cl/cl.shift(5)-1)/r.rolling(20,min_periods=15).std()).shift(1)
for h in [1,3,5,10]:
 fut=cl.shift(-h)/cl-1; q=[]; ns=[]
 for dt in cl.index:
  z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   a=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(a):q.append(a);ns.append(len(z))
 q=pd.Series(q); print('H',h,'obs',len(q),'IC',round(q.mean(),7),'ICIR',round(q.mean()/q.std(ddof=1)*np.sqrt(len(q)),4),'hit',round((q>0).mean(),4),'avg_n',round(np.mean(ns),2),'recent500',round(q.tail(500).mean(),7))
print('cutoff',cl.index.max().date(),'dates',len(cl),'instruments',len(cl.columns),'coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20271230_volscaled_reversal5_signal.csv',index=False)
