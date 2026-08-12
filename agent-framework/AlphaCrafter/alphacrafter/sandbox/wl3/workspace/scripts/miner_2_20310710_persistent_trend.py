import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<200: d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  q=d[['date','close']].drop_duplicates('date'); q['symbol']=s; rows.append(q)
x=pd.concat(rows).pivot(index='date',columns='symbol'); cl=x['close'].sort_index().ffill(); r=cl.pct_change()
# Smooth trend persistence: lagged 10D return, penalized by realized volatility and disagreement of daily signs.
# Signal is lagged one session; forward returns are never used in construction.
rv=r.rolling(20,min_periods=15).std(); consistency=r.rolling(10,min_periods=8).mean()/(r.rolling(10,min_periods=8).std()+1e-12)
f=(cl.pct_change(10)*consistency/(rv+1e-12)).shift(1)
def calc(h, dates):
 a=[]; ns=[]
 for dt in dates:
  z=pd.concat([f.loc[dt],(cl.shift(-h).loc[dt]/cl.loc[dt]-1)],axis=1).dropna()
  if len(z)>=8:
   v=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(v): a.append(v); ns.append(len(z))
 q=pd.Series(a)
 return len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean(),np.mean(ns)
for h in [1,3,5,10,20]: print('horizon',h,calc(h,cl.index))
for label,mask in [('2020-22',cl.index<'2023-01-01'),('2023-25',(cl.index>='2023-01-01')&(cl.index<'2026-01-01')),('2026-29',(cl.index>='2026-01-01')&(cl.index<'2030-01-01')),('2030+',cl.index>='2030-01-01'),('recent120',np.arange(len(cl))>=len(cl)-120)]: print('regime',label,calc(10,cl.index[mask]))
print('cutoff',cl.index.max().date(),'dates',len(cl),'instruments',len(cl.columns),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20310710_persistent_trend_signal.csv',index=False)
