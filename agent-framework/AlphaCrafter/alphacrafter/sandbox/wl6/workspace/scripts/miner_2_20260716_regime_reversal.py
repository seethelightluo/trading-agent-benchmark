import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for a in A:
 try:
  x=get_stock_daily_data(a,days=4000)
  if x is not None:D[a]=x.set_index('date').close.astype(float)
 except:pass
p=pd.concat(D,axis=1,sort=True).ffill();r=p.pct_change();rev=-(p/p.shift(3)-1); vol=r.rolling(20,min_periods=10).std(); f=rev/(vol*np.sqrt(3)); rv=r.mean(axis=1).rolling(20).std(); med=rv.rolling(252,min_periods=60).median()
for label,gate in [('low',rv<=med),('high',rv>med),('extreme_low',rv<=rv.rolling(504,min_periods=120).quantile(.3)),('calm',rv<=rv.rolling(252,min_periods=60).quantile(.4))]:
 vals=[]; ncs=[]
 for i,dt in enumerate(p.index[:-1]):
  if not gate.loc[dt]:continue
  q=pd.concat([f.loc[dt],(p.iloc[i+1]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8:
   vals.append(q.iloc[:,0].corr(q.y));ncs.append(len(q))
 s=pd.Series(vals);print(label,'dates',len(s),'coverage_dates',round(len(s)/len(p),4),'avgN',round(np.mean(ncs),2),'IC',round(s.mean(),5),'ICIR',round(s.mean()/s.std(),5),'hit',round((s>0).mean(),4))
