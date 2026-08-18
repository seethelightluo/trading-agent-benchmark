import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];o={}
for s in U:
 d=get_stock_daily_data(s,days=6000)
 if d is None or len(d)<300:d=get_index_daily_data(s,days=6000)
 o[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(o).sort_index().ffill(); fac=p.shift(1)/p.shift(21)-1; fwd=p.shift(-20)/p-1
for name,x in [('mom20',fac),('reversal20',-fac)]:
 vals=[];ns=[]
 for dt in x.index:
  z=pd.concat([x.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=pd.Series(vals).dropna();print(name,len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(len(a)),(a>0).mean(),x.notna().sum().sum()/(len(x)*15))
