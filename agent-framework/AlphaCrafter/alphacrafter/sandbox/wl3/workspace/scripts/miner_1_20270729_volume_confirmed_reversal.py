import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,2800)
 if d is None or len(d)<200: d=get_index_daily_data(s,2800)
 if d is not None and len(d):
  x=d[['date','close','volume']].drop_duplicates('date'); x['symbol']=s; rows.append(x)
p=pd.concat(rows); c=p.pivot(index='date',columns='symbol',values='close').sort_index().ffill(); v=p.pivot(index='date',columns='symbol',values='volume').reindex(c.index).ffill(); r=c.pct_change()
# Volume-confirmed short-horizon reversal: recent losses with unusually high trading activity
ret10=c/c.shift(10)-1
vmean=v.rolling(60,min_periods=30).mean(); vstd=v.rolling(60,min_periods=30).std(); vz=(v-vmean)/vstd
f=(-ret10)*(1+0.25*np.tanh(vz/2))
f=f.sub(f.median(axis=1),axis=0).clip(-6,6)
def calc(h):
 fut=c.shift(-h)/c-1; qs=[];ns=[]
 for dt in c.index:
  z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):qs.append(q);ns.append(len(z))
 q=pd.Series(qs);return len(q),q.mean(),q.std(ddof=1),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean(),np.mean(ns)
print('cutoff',c.index.max().date(),'dates',len(c),'instruments',len(c.columns))
for h in [1,3,5,10]:print('H',h,calc(h))
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01',str(c.index.max().date()))]:
 fut=c.shift(-1)/c-1;qs=[]
 for dt in c.loc[a:b].index:
  z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8:qs.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(qs);print('REG',a,b,len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)))
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20270729_volume_confirmed_reversal_signal.csv',index=False)
print('max_abs_library_correlation',None)
