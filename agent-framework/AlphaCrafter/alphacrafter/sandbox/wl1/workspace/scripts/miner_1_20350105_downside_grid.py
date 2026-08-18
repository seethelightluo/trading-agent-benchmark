import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def L(s):
 d=get_stock_daily_data(s,6000)
 if d is None or len(d)<100:d=get_index_daily_data(s,6000)
 return d.set_index(pd.to_datetime(d.date)).close.rename(s)
px=pd.concat([L(s) for s in U],axis=1).sort_index().ffill(); r=px.pct_change()
for w in [10,20,40]:
 dv=r.where(r<0,0).rolling(30,min_periods=20).std()*np.sqrt(252)
 sig=px.pct_change(w).div(dv).shift(1)
 print('WINDOW',w)
 for h in [5,10,20,40]:
  f=px.shift(-h).div(px)-1; a=[]; ns=[]
  for dt in sig.index:
   z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
   if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]));ns.append(len(z))
  a=pd.Series(a).dropna();print(h,len(a),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6),round(np.mean(ns),2))
