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
z=pd.concat(rows); cl=z.pivot(index='date',columns='symbol',values='close').sort_index().ffill(); hi=z.pivot(index='date',columns='symbol',values='high').reindex(cl.index).ffill(); lo=z.pivot(index='date',columns='symbol',values='low').reindex(cl.index).ffill()
# Distance from 20-day range midpoint, volatility-normalized; contrarian signal, lagged.
mid=(hi.rolling(20).max()+lo.rolling(20).min())/2
rng=(hi.rolling(20).max()-lo.rolling(20).min()).replace(0,np.nan)
f=(-(cl-mid)/rng).shift(1)
def run(h,ix):
 a=[]
 for dt in ix:
  q=pd.concat([f.loc[dt],cl.shift(-h).loc[dt]/cl.loc[dt]-1],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): a.append(v)
 q=pd.Series(a);return len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean()
for h in [1,3,5,10]: print('horizon',h,run(h,cl.index))
print('meta',cl.index.max().date(),len(cl),len(cl.columns),f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for n,ix in [('2020-22',cl.index<'2023-01-01'),('2023-25',(cl.index>='2023-01-01')&(cl.index<'2026-01-01')),('2026+',cl.index>='2026-01-01'),('recent120',cl.index>=cl.index[-120])]: print(n,run(5,cl.index[ix]))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20281130_range_mid_reversal20_signal.csv',index=False)
