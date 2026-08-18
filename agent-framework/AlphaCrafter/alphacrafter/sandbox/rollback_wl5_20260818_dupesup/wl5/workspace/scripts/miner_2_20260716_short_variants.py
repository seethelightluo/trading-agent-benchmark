import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for a in A:
 try:
  x=get_stock_daily_data(a,days=4000)
  if x is not None:D[a]=x.set_index('date').close.astype(float)
 except:pass
p=pd.concat(D,axis=1,sort=True).ffill();
for lb in [1,2,4,6,7,8]:
 f=-(p/p.shift(lb)-1); vals=[]
 for i,dt in enumerate(p.index[:-1]):
  q=pd.concat([f.loc[dt],(p.iloc[i+1]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8: vals.append(q.iloc[:,0].corr(q.y))
 s=pd.Series(vals);print('lb',lb,'dates',len(s),'IC',round(s.mean(),5),'ICIR',round(s.mean()/s.std(),5),'hit',round((s>0).mean(),4),'turn',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'corr3',round(pd.concat([f.stack(),(-(p/p.shift(3)-1)).stack()],axis=1).dropna().corr().iloc[0,1],4))
