import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300:d=get_index_daily_data(s,4000)
 if d is not None:D[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=np.log(p).diff(); vol=r.rolling(40,min_periods=25).std()*np.sqrt(252)
for br in [.55,.60,.65,.70]:
 for look in [10,20]:
  f=r.rolling(look).sum().div(vol).mul((r.rolling(5).sum().gt(0).mean(axis=1)>br).astype(float),axis=0)
  rows=[]
  for t in f.index:
   j=r.index.searchsorted(t,side='right'); k=j+9
   if k>=len(r):continue
   z=pd.concat([f.loc[t],r.iloc[j:k+1].sum()],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].std()>0:rows.append((t,z.iloc[:,0].corr(z.iloc[:,1])))
  x=pd.Series(dict(rows)); recent=x[x.index.year>=2028]
  print('BR',br,'L',look,'n',len(x),'recent',len(recent),'IC',round(x.mean(),5),'ICIR',round(x.mean()/x.std(),5),'recentIC',round(recent.mean(),5),'recentIR',round(recent.mean()/recent.std(),5))
