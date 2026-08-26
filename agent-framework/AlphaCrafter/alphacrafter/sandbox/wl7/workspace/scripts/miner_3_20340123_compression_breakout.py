import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:x=fn(s,days=5000)
  except:pass
  if x is not None and len(x):break
 if x is not None:D[s]=x.set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index().ffill(); R=P.pct_change(); vol=R.rolling(20).std(); med=vol.median(axis=1)
# low-vol breakout: prior 20d momentum gated by vol below cross-sectional median, all lagged one day
for k in [0.75,1.0]:
 f=(P.shift(1)/P.shift(21)-1).where(vol.shift(1).div(med.shift(1),axis=0)<k)
 f=f.sub(f.median(axis=1),axis=0); print('gate',k,'coverage',f.notna().mean().mean())
 for h in [5,10,20]:
  q=[];ns=[]
  for i in range(len(P)-h):
   z=pd.concat([f.iloc[i],(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,:-1].nunique().iloc[0]>2:q.append(z.iloc[:,0].corr(z.y));ns.append(len(z))
  q=pd.Series(q).dropna();print('H',h,'dates',len(q),'N',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
