import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 try:d=get_index_daily_data(s,3000)
 except FileNotFoundError:d=get_stock_daily_data(s,3000)
 if d is None:return pd.DataFrame()
 d=d.copy();d.date=pd.to_datetime(d.date);return d.set_index('date')
ds={s:load(s) for s in U}; close=pd.concat({s:x.close for s,x in ds.items()},axis=1).sort_index().ffill()
open_=pd.concat({s:x.open for s,x in ds.items()},axis=1).reindex(close.index).ffill(); high=pd.concat({s:x.high for s,x in ds.items()},axis=1).reindex(close.index).ffill(); low=pd.concat({s:x.low for s,x in ds.items()},axis=1).reindex(close.index).ffill()
r=np.log(close).diff(); atr=(high-low).rolling(14).mean(); # intraday reversal, normalized by range
f=(-(open_/close-1)/(atr/close)).shift(1).replace([np.inf,-np.inf],np.nan)
ics=[];ns=[]; dates=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],r.shift(-1).loc[dt]],axis=1).dropna()
 if len(z)>=8:ics.append(z.iloc[:,0].corr(z.iloc[:,1]));ns.append(len(z));dates.append(dt)
s=pd.Series(ics,index=pd.to_datetime(dates));print('dates',len(s),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15);print('IC %.8f ICIR %.8f hit %.4f turnover %.4f'%(s.mean(),s.mean()/s.std()*np.sqrt(252),(s>0).mean(),f.rank(axis=1,pct=True).diff().abs().mean().mean()))
for h in [5,10,20]:
 rr=r.shift(-h).rolling(h).sum().shift(-(h-1));a=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],rr.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('decay',h,np.nanmean(a))
for a,b in [('2020','2022'),('2023','2024'),('2025','2027')]:print('regime',a,b,s[(s.index>=a)&(s.index<=b+'-12-31')].mean())
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20270303_intraday_range_reversal_signal.csv',index=False)
