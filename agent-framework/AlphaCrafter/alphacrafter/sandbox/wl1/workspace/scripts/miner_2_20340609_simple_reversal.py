import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=get_stock_daily_data(s,5000)
 if x is None or len(x)==0:x=get_index_daily_data(s,5000)
 if x is not None and len(x):D[s]=x.assign(date=pd.to_datetime(x.date)).set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change();vol=r.rolling(20,min_periods=15).std();
for look in [10,20]:
 f=(-p.pct_change(look)/vol).shift(1); y=p.shift(-10)/p-1; z=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(a)>=8:z.append(a.iloc[:,0].corr(a.iloc[:,1]))
 q=pd.Series(z).dropna();print('look',look,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(),'hit',(q>0).mean())
 print('decay',[(h, np.nanmean([pd.concat([f.loc[d],(p.shift(-h)/p-1).loc[d]],axis=1).dropna().iloc[:,0].corr(pd.concat([f.loc[d],(p.shift(-h)/p-1).loc[d]],axis=1).dropna().iloc[:,1]) for d in f.index])) for h in [5,10,20]])
