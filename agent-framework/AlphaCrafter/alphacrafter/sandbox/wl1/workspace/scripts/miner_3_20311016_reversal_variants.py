import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x)>80:x=x[['date','close']];x.date=pd.to_datetime(x.date);D[s]=x.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(D).sort_index().ffill();r=np.log(p).diff();v=r.rolling(20).std();
for k in [0,0.25,0.5,0.75,1.0]:
 disp=r.sub(r.mean(axis=1),axis=0).rolling(10).std().mean(axis=1); damp=(1+disp/disp.rolling(120).median())**(-k);f=(-np.log(p/p.shift(3))/(v+1e-8)).mul(damp,axis=0);q=[]
 for i in range(len(p)-1):
  z=pd.concat([f.iloc[i],r.iloc[i+1]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(q).dropna();print(k,len(q),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean())
