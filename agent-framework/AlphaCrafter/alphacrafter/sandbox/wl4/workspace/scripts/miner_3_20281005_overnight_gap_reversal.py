import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; ds={}
for s in U:
 d=get_stock_daily_data(s,2600)
 if d is None or len(d)<300:d=get_index_daily_data(s,2600)
 if d is not None:ds[s]=d.set_index('date')
op=pd.concat({s:x['open'] for s,x in ds.items()},axis=1).sort_index().ffill(); cl=pd.concat({s:x['close'] for s,x in ds.items()},axis=1).sort_index().ffill()
# gap versus prior close, lag decision
f=-(op/cl.shift(1)-1).shift(1)
print('rows',len(cl),'assets',len(ds),'coverage',round(f.notna().mean().mean(),4))
for h in [1,5,10,20]:
 y=cl.pct_change(h).shift(-h);a=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(a);a=a[np.isfinite(a)];print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round((a>0).mean(),4))
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),5))
