import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,days=2500)
 if d is not None:D[s]=d.sort_values('date').set_index('date')
p=pd.concat({s:d['close'] for s,d in D.items()},axis=1).sort_index(); o=pd.concat({s:d['open'] for s,d in D.items()},axis=1).reindex(p.index); c=p
# factor: intraday return reversal, prior day's open-close, and 5d average
intr=c/o-1
for w in [1,3,5,10]:
 fac=-intr.rolling(w,min_periods=w).mean(); a=[]; ns=[]
 for i in range(w,len(p)-1):
  z=pd.concat([fac.iloc[i],(p.iloc[i+1]/p.iloc[i]-1)],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(a); print(w,len(a),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0),np.mean(ns))
