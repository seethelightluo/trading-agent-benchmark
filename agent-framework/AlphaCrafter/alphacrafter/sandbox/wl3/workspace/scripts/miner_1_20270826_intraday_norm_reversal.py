import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<200:d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  x=d[['date','open','close','high','low']].drop_duplicates('date');x['symbol']=s;rows.append(x)
p=pd.concat(rows); cols=['open','close','high','low']; q={z:p.pivot(index='date',columns='symbol',values=z).sort_index().ffill() for z in cols}; c=q['close']
atr=(q['high']-q['low']).rolling(14,min_periods=10).mean(); intr=(q['close']/q['open']-1)/atr.replace(0,np.nan)
# Reversal of normalized same-session move, with a modest recent-trend filter
trend=c/c.shift(5)-1
f=(-intr)*(1+0.25*np.tanh((-trend)/0.05)); f=f.sub(f.median(axis=1),axis=0).clip(-6,6)
def calc(h,ix=None):
 fut=c.shift(-h)/c-1; qs=[];ns=[]
 for dt in (c.index if ix is None else ix):
  z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   x=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(x):qs.append(x);ns.append(len(z))
 x=pd.Series(qs);return len(x),x.mean(),x.std(ddof=1),x.mean()/x.std(ddof=1)*np.sqrt(len(x)),(x>0).mean(),np.mean(ns)
print('cutoff',c.index.max().date(),'dates',len(c),'instruments',len(c.columns))
for h in [1,3,5,10]:print('H',h,calc(h))
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01',str(c.index.max().date()))]:
 n,ic,sd,ir,hit,avg=calc(1,c.loc[a:b].index);print('REG',a,b,'n',n,'ic',ic,'ir',ir,'hit',hit,'avgN',avg)
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20270826_intraday_norm_reversal_signal.csv',index=False)
print('max_abs_library_correlation',None)
