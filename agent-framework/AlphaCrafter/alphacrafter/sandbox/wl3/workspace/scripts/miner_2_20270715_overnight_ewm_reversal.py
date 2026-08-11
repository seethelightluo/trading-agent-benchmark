import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=get_stock_daily_data(s,2800)
 if d is None or len(d)<200:d=get_index_daily_data(s,2800)
 if d is not None and len(d):
  z=d[['date','open','close']].drop_duplicates('date');z['symbol']=s;rows.append(z)
a=pd.concat(rows); c=a.pivot(index='date',columns='symbol',values='close').sort_index().ffill(); o=a.pivot(index='date',columns='symbol',values='open').reindex(c.index).ffill()
g=o/c.shift(1)-1; rv=c.pct_change().rolling(30,min_periods=20).std(); raw=-g/(rv+1e-12); f=raw.ewm(span=3,min_periods=3).mean(); f=f.sub(f.median(axis=1),axis=0).clip(-6,6)
def ev(fr):
 qs=[];ns=[];ds=[]
 for dt in c.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):qs.append(q);ns.append(len(z));ds.append(dt)
 q=pd.Series(qs,index=pd.DatetimeIndex(ds));return len(q),np.mean(ns),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean(),q
print('cutoff',c.index.max().date(),'dates',len(c),'instruments',len(c.columns))
for h in [1,3,5,10]:
 r=ev(c.shift(-h)/c-1);print('H',h,'obs',r[0],'avg_n',r[1],'IC',r[2],'ICIR',r[3],'hit',r[4])
r=ev(c.shift(-1)/c-1)
for a1,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01',str(c.index.max().date()))]:
 q=r[5].loc[a1:b];print('REG',a1,b,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(len(q)))
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20270715_overnight_ewm_reversal_signal.csv',index=False)
