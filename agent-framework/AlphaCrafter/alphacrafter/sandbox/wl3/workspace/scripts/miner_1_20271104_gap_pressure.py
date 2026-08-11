import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<200:d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  x=d[['date','open','close']].drop_duplicates('date');x['symbol']=s;rows.append(x)
p=pd.concat(rows)
op=p.pivot(index='date',columns='symbol',values='open').sort_index().ffill(); cl=p.pivot(index='date',columns='symbol',values='close').reindex(op.index).ffill()
# Close-to-open gap is known after session t; use lagged 3-session mean and volatility scaling.
gap=op/cl.shift(1)-1
rv=gap.rolling(20).std().replace(0,np.nan)
f=(gap.rolling(3).mean()/rv).shift(1)
def calc(h):
 fut=cl.shift(-h)/cl-1;q=[];ns=[]
 for dt in cl.index:
  z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   v=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(v):q.append(v);ns.append(len(z))
 q=pd.Series(q);return len(q),q.mean(),q.std(ddof=1),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean(),np.mean(ns)
print('cutoff',cl.index.max().date(),'dates',len(cl),'instruments',len(cl.columns))
for h in [1,3,5,10]:print('H',h,calc(h))
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20271104_gap_pressure_signal.csv',index=False)
