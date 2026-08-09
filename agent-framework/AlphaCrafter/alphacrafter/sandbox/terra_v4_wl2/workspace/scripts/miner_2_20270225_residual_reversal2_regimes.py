import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None:return x
  except:pass
px=pd.DataFrame({s:g(s).set_index('date')['close'] for s in U}).sort_index();r=px.pct_change();fr=px.shift(-1)/px-1
for look in [2]:
 raw=r.rolling(look).sum();f=-(raw.sub(raw.median(axis=1),axis=0));f=f.sub(f.median(axis=1),axis=0)
 for label,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027','2027','2027-12-31')]:
  a=[];ns=[]
  for dt in f.loc[lo:hi].index:
   z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
  a=np.array(a);print(label,len(a),round(np.mean(ns),2) if len(ns) else 0,round(a.mean(),6) if len(a) else None,round(a.mean()/a.std(ddof=1),6) if len(a)>1 else None,round((a>0).mean(),4) if len(a) else None)
