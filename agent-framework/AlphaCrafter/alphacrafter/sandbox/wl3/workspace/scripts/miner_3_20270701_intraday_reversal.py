import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,2800)
 if d is None or len(d)<200:d=get_index_daily_data(s,2800)
 if d is not None and len(d):
  x=d[['date','open','close']].drop_duplicates('date');x['symbol']=s;rows.append(x)
raw=pd.concat(rows).pivot(index='date',columns='symbol',values='close').sort_index().ffill()
op=pd.concat(rows).pivot(index='date',columns='symbol',values='open').reindex(raw.index).ffill()
intra=raw/op-1
# Reversal of the prior completed session's intraday move, scaled by 30d close volatility.
vol=raw.pct_change().rolling(30,min_periods=20).std()
base=-intra/(vol+1e-12)
f=base.sub(base.median(axis=1),axis=0).clip(-6,6)
fut=raw.shift(-1)/raw-1
qs=[];dates=[];ns=[]
for dt in raw.index:
 z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):qs.append(q);dates.append(dt);ns.append(len(z))
q=pd.Series(qs,index=pd.DatetimeIndex(dates))
print('cutoff',raw.index.max().date(),'dates',len(raw),'instruments',len(raw.columns),'ic_dates',len(q),'avg_n',np.mean(ns))
print('IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(len(q)),'hit',(q>0).mean())
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01',str(raw.index.max().date()))]:
 z=q.loc[a:b];print('REG',a,b,'n',len(z),'ic',z.mean(),'icir',z.mean()/z.std(ddof=1)*np.sqrt(len(z)) if len(z)>1 else np.nan)
for h in [1,3,5,10]:
 fr=raw.shift(-h)/raw-1;qq=[]
 for dt in raw.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: qq.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 qq=pd.Series(qq).dropna();print('DECAY',h,'ic',qq.mean(),'icir',qq.mean()/qq.std(ddof=1)*np.sqrt(len(qq)))
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20270701_intraday_reversal_signal.csv',index=False)
print('max_abs_library_correlation',None)
