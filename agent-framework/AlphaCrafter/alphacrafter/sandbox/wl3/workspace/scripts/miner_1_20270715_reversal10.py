import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,2800)
 if d is None or len(d)<200: d=get_index_daily_data(s,2800)
 if d is not None and len(d):
  x=d[['date','close']].drop_duplicates('date'); x['symbol']=s; rows.append(x)
w=pd.concat(rows).pivot(index='date',columns='symbol',values='close').sort_index().ffill()
r=w.pct_change()
# Volatility-normalized 10-session cross-asset reversal, with robust row-wise clipping.
raw=-w.pct_change(10)/(r.rolling(20,min_periods=15).std()*np.sqrt(20)+1e-12)
f=raw.sub(raw.median(axis=1),axis=0).clip(-6,6)
# forward horizons are strictly after the signal date
metrics={}
for h in [1,3,5,10]:
 fut=w.shift(-h)/w-1; qs=[]; ns=[]; ds=[]
 for dt in w.index:
  z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): qs.append(q); ns.append(len(z)); ds.append(dt)
 q=pd.Series(qs,index=pd.DatetimeIndex(ds)); metrics[h]=q
 print('H',h,'ic_dates',len(q),'avg_n',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1)*np.sqrt(len(q)),6),'hit',round((q>0).mean(),4))
 if h==1:
  for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01',str(w.index.max().date()))]:
   z=q.loc[a:b]; print('REG',a,b,'n',len(z),'ic',round(z.mean(),6),'icir',round(z.mean()/z.std(ddof=1)*np.sqrt(len(z)),6) if len(z)>1 else np.nan)
q=metrics[1]
print('cutoff',w.index.max().date(),'dates',len(w),'instruments',len(w.columns),'coverage',round(f.notna().mean().mean(),6),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6),'max_abs_library_correlation',None)
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20270715_reversal10_signal.csv',index=False)
