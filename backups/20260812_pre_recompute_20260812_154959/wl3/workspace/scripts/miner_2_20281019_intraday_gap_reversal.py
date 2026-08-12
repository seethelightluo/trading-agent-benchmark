import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<200: d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  z=d[['date','open','close']].drop_duplicates('date');z['symbol']=s;rows.append(z)
x=pd.concat(rows).pivot(index='date',columns='symbol',values=['open','close']).sort_index().ffill()
op,cl=x['open'],x['close']
f=(-(cl/op-1)).shift(1)
def calc(h,mask=None):
 a=[];ix=cl.index if mask is None else cl.index[mask];future=cl.shift(-h)/cl-1
 for dt in ix:
  q=pd.concat([f.loc[dt],future.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   r=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(r):a.append(r)
 q=pd.Series(a);return len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean()
for h in [1,3,5,10]:print('horizon',h,'obs IC ICIR hit',calc(h))
print('cutoff',cl.index.max().date(),'dates',len(cl),'instruments',len(cl.columns),'avgN',f.notna().sum(axis=1).mean(),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for name,mask in [('2020-22',cl.index<'2023-01-01'),('2023-25',(cl.index>='2023-01-01')&(cl.index<'2026-01-01')),('2026+',cl.index>='2026-01-01'),('recent120',cl.index>=cl.index[-120])]:print('regime',name,calc(1,mask))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20281019_intraday_gap_reversal_signal.csv',index=False)
