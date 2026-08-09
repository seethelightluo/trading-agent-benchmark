import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None:return x
  except: pass
px=pd.DataFrame({s:g(s).set_index('date')['close'] for s in U}).sort_index(); r=px.pct_change(); fr=px.shift(-1)/px-1
b=(r.lt(0).sum(axis=1)/r.notna().sum(axis=1)).shift(1)
for t in [.60,.70,.80,.90]:
 active=b>=t; f=(-r.rolling(3).sum()).where(active); f=f.sub(f.median(axis=1),axis=0);a=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.asarray(a); print('threshold',t,'dates',len(a),'avgN',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1) if len(a)>1 else np.nan,'hit',np.mean(a>0),'coverage',f.notna().sum().sum()/len(U)/len(f))
