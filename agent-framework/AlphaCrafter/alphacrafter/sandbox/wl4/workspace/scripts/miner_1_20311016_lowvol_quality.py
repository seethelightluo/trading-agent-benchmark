import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=3000)
 if x is not None and len(x): D[s]=x.assign(date=pd.to_datetime(x.date)).set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index().ffill(); r=np.log(P).diff()
# Low-volatility quality: inverse realized volatility, mildly rewarded for stable positive breadth.
vol=r.rolling(40,min_periods=25).std(); breadth=(r>0).rolling(20,min_periods=15).mean()
f=(1/(vol+1e-8)*(0.5+breadth)).shift(1)
print('universe',len(D),'dates',len(P),'end',P.index.max().date())
for h in [5,10,20]:
 q=[];ns=[]
 for i in range(45,len(P)-h):
  z=pd.concat([f.iloc[i],np.log(P.iloc[i+h]/P.iloc[i])],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 q=pd.Series(q).dropna();print('H',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'coverage',round(P.notna().mean().mean(),4))
 for w in [365,730]:
  x=q.tail(w);print('recent',w,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'n',len(x))
