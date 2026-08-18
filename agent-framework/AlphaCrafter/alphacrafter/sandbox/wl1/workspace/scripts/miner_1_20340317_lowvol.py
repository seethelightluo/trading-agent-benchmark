import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0:d=get_index_daily_data(s,5000)
 if d is not None and len(d):P[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
px=pd.DataFrame(P).sort_index().ffill();r=px.pct_change()
# low-volatility anomaly: inverse realized volatility, optionally conditioned on trend
for w in [10,20,40,60]:
 for h in [10,20]:
  f=(-r.rolling(w,min_periods=max(10,w//2)).std()).shift(1); fr=px.pct_change(h).shift(-h); z=[];ns=[]
  for dt in f.index:
   a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
   if len(a)>=8:z.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman'));ns.append(len(a))
  z=np.array(z); print('w',w,'h',h,'dates',len(z),'N',np.mean(ns),'IC',np.nanmean(z),'ICIR',np.nanmean(z)/np.nanstd(z,ddof=1),'hit',np.mean(z>0))
