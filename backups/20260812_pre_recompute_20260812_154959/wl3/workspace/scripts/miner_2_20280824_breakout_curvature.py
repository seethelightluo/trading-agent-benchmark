import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<200: d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  x=d[['date','close','high','low']].drop_duplicates('date'); x['symbol']=s; rows.append(x)
cl=pd.concat([x[['date','symbol','close']] for x in rows]).pivot(index='date',columns='symbol',values='close').sort_index().ffill()
hi=pd.concat([x[['date','symbol','high']] for x in rows]).pivot(index='date',columns='symbol',values='high').reindex(cl.index).ffill()
lo=pd.concat([x[['date','symbol','low']] for x in rows]).pivot(index='date',columns='symbol',values='low').reindex(cl.index).ffill()
rng=((hi-lo)/cl).rolling(10,min_periods=7).mean()
d20=(cl/cl.rolling(20,min_periods=15).max()-1)/rng
d60=(cl/cl.rolling(60,min_periods=40).max()-1)/rng
# Curvature: short breakout pressure in excess of longer lookback pressure; lagged one session.
f=(d20-d60).shift(1)
def calc(h, dates):
 a=[]; ns=[]
 for dt in dates:
  z=pd.concat([f.loc[dt],cl.shift(-h).loc[dt]/cl.loc[dt]-1],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): a.append(q); ns.append(len(z))
 q=pd.Series(a)
 return len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean(),np.mean(ns)
for h in [1,3,5,10]: print('horizon',h,'obs IC ICIR hit avgN',calc(h,cl.index))
for label,mask in [('2020-22',cl.index<'2023-01-01'),('2023-25',(cl.index>='2023-01-01')&(cl.index<'2026-01-01')),('2026+',cl.index>='2026-01-01'),('recent120',np.arange(len(cl))>=len(cl)-120)]: print('regime',label,'h10',calc(10,cl.index[mask]))
print('cutoff',cl.index.max().date(),'dates',len(cl),'instruments',len(cl.columns),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
f.stack().rename('signal').reset_index().to_csv('scripts/miner_2_20280824_breakout_curvature_signal.csv',index=False)
