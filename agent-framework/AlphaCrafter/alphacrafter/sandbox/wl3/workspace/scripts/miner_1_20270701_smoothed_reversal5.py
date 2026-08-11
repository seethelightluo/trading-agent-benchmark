import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,2800)
 if d is None or len(d)<200:d=get_index_daily_data(s,2800)
 if d is not None and len(d):
  x=d[['date','close']].drop_duplicates('date'); x['symbol']=s; rows.append(x)
w=pd.concat(rows).pivot(index='date',columns='symbol',values='close').sort_index().ffill(); r=w.pct_change()
base=-w.pct_change(5)/(r.rolling(20,min_periods=12).std()*np.sqrt(20)+1e-12)
# Three-session exponentially weighted smoothing reduces rank churn while preserving short reversal.
f=base.ewm(span=3,min_periods=2,adjust=False).mean()
f=f.sub(f.median(axis=1),axis=0).clip(-6,6)
fut=w.shift(-1)/w-1
qs=[]; dates=[]; ns=[]
for dt in w.index:
 z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q): qs.append(q);dates.append(dt);ns.append(len(z))
q=pd.Series(qs,index=pd.DatetimeIndex(dates));
print('cutoff',w.index.max().date(),'dates',len(w),'instruments',len(w.columns),'ic_dates',len(q),'avg_n',round(np.mean(ns),2))
print('IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(len(q)),'hit',(q>0).mean())
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01',str(w.index.max().date()))]:
 z=q.loc[a:b]; print('REG',a,b,'n',len(z),'ic',z.mean(),'icir',z.mean()/z.std(ddof=1)*np.sqrt(len(z)) if len(z)>1 else np.nan)
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for h in [1,3,5,10]:
 fr=w.shift(-h)/w-1; qq=[]
 for dt in w.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: qq.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('DECAY',h,np.nanmean(qq),len(qq))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20270701_smoothed_reversal5_signal.csv',index=False)
print('max_abs_library_correlation',None)
