import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is not None and len(d):
  d=d.copy();d.date=pd.to_datetime(d.date);D[s]=d.sort_values('date').set_index('date').close
P=pd.concat(D,axis=1).sort_index().ffill(); r=P.pct_change(); r10=P/P.shift(10)-1
breadth=(r10>0).mean(axis=1)
for qv in [.75,.8,.85,.9]:
 q=breadth.shift(1).rolling(120,min_periods=60).quantile(qv); cond=breadth.shift(1)>=q
 for direction in [-1,1]:
  sig=direction*r10.where(cond,np.nan);sig=sig.sub(sig.median(axis=1),axis=0);f=P.shift(-1)/P-1; vals=[]
  for dt in sig.index:
   z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
   if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
  a=np.array(vals); print(qv,direction,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'cov',sig.notna().mean().mean())
