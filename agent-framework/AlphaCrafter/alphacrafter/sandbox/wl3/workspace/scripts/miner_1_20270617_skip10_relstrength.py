import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,2600)
 if d is None or len(d)<200: d=get_index_daily_data(s,2600)
 if d is not None and len(d):
  x=d[['date','close']].drop_duplicates('date'); x['symbol']=s; rows.append(x)
w=pd.concat(rows).pivot(index='date',columns='symbol',values='close').sort_index().ffill()
r=w.pct_change(); vol=r.rolling(60,min_periods=40).std()
# medium-term relative strength, skip most recent 10 sessions
f=(w.shift(10)/w.shift(70)-1)/(vol*np.sqrt(60)+1e-12)
f=f.sub(f.median(axis=1),axis=0).clip(-8,8)
fut=w.shift(-1)/w-1
qs=[]; dates=[]; ns=[]
for dt in w.index:
 z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q): qs.append(q); dates.append(dt); ns.append(len(z))
q=pd.Series(qs,index=pd.DatetimeIndex(dates))
print('cutoff',w.index.max().date(),'dates',len(w),'instruments',len(w.columns),'ic_dates',len(q),'avg_n',round(np.mean(ns),2))
for h in [1,3,5,10]:
 ff=w.shift(-h)/w-1; vals=[]; ds=[]
 for dt in w.index:
  z=pd.concat([f.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   v=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(v): vals.append(v);ds.append(dt)
 s=pd.Series(vals,index=pd.DatetimeIndex(ds)); print('H',h,'n',len(s),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1)*np.sqrt(len(s)),4),'hit',round((s>0).mean(),4))
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01',str(w.index.max().date()))]:
 z=q.loc[a:b];print('REG',a,b,'n',len(z),'ic',round(z.mean(),6),'icir',round(z.mean()/z.std(ddof=1)*np.sqrt(len(z)),4))
print('coverage',round(f.notna().mean().mean(),6),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20270617_skip10_relstrength_signal.csv',index=False)
print('max_abs_library_correlation',None)
