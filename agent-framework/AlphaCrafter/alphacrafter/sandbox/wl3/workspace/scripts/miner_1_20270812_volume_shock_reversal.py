import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<200: d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  x=d[['date','close','volume']].drop_duplicates('date'); x['symbol']=s; rows.append(x)
p=pd.concat(rows); c=p.pivot(index='date',columns='symbol',values='close').sort_index().ffill(); v=p.pivot(index='date',columns='symbol',values='volume').reindex(c.index).ffill()
# Candidate: short reversal intensified only by exceptional activity, with robust percentile-like z score
ret3=c/c.shift(3)-1
vm=v.rolling(60,min_periods=30).mean(); vs=v.rolling(60,min_periods=30).std(); vz=((v-vm)/vs).replace([np.inf,-np.inf],np.nan)
f=(-ret3)*(1+0.35*np.tanh(vz/2))
f=f.sub(f.median(axis=1),axis=0).clip(-6,6)
def calc(h,ix=None):
 fut=c.shift(-h)/c-1; qs=[];ns=[]
 dates=c.index if ix is None else ix
 for dt in dates:
  z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):qs.append(q);ns.append(len(z))
 q=pd.Series(qs);return len(q),q.mean(),q.std(ddof=1),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean(),np.mean(ns)
print('cutoff',c.index.max().date(),'dates',len(c),'instruments',len(c.columns))
for h in [1,3,5,10]: print('H',h,calc(h))
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01',str(c.index.max().date()))]:
 n,ic,sd,ir,hit,avg=calc(1,c.loc[a:b].index);print('REG',a,b,n,avg,ir,hit, 'avgN',avg)
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20270812_volume_shock_reversal_signal.csv',index=False)
print('max_abs_library_correlation',None)
