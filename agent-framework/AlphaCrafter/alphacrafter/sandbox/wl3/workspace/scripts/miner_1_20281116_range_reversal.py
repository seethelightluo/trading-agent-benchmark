import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<200: d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  x=d[['date','close','high','low']].drop_duplicates('date');x['symbol']=s;rows.append(x)
z=pd.concat(rows).pivot(index='date',columns='symbol'); z=z.sort_index().ffill()
cl=z['close']; hi=z['high']; lo=z['low']; r=cl.pct_change()
# Range-adjusted short reversal: reverse recent 3d move, scaled by typical true range, lagged one day.
prev=cl.shift(1); tr=pd.concat([(hi-lo)/prev,(hi-prev).abs()/prev,(lo-prev).abs()/prev],axis=1).groupby(level=0,axis=1).max() if False else None
# compute per symbol straightforward
atr=(hi-lo).div(cl).rolling(20,min_periods=15).mean()
f=(-cl.pct_change(3)/atr).shift(1)
def calc(h,mask=None):
 out=[]; dates=cl.index if mask is None else cl.index[mask]
 for dt in dates:
  q=pd.concat([f.loc[dt],(cl.shift(-h).loc[dt]/cl.loc[dt]-1)],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): out.append(v)
 q=pd.Series(out); return len(q),float(q.mean()),float(q.mean()/q.std(ddof=1)*np.sqrt(len(q))),float((q>0).mean())
for h in [1,3,5,10]: print('horizon',h,'obs IC ICIR hit',calc(h))
print('cutoff',cl.index.max().date(),'dates',len(cl),'instruments',len(cl.columns),'avgN',f.notna().sum(axis=1).mean(),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for name,mask in [('2020-22',cl.index<'2023-01-01'),('2023-25',(cl.index>='2023-01-01')&(cl.index<'2026-01-01')),('2026+',cl.index>='2026-01-01'),('recent120',cl.index>=cl.index[-120])]: print('regime',name,calc(1,mask),calc(5,mask))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20281116_range_reversal_signal.csv',index=False)
