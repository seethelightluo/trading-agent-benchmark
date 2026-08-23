import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-01-27')
def g(s):
 for f in(get_index_daily_data,get_stock_daily_data):
  try:
   x=f(s,3000)
   if x is not None and len(x):
    x=x.copy();x.date=pd.to_datetime(x.date).dt.normalize();return x.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except:pass
D={s:g(s) for s in U};D={s:x for s,x in D.items() if x is not None}; C=pd.concat({s:x.close for s,x in D.items()},axis=1)
r20=C.pct_change(20);r60=C.pct_change(60);v=C.pct_change().rolling(40).std();F=((r20-r60/3)/v).shift(1); FR=C.shift(-1)/C-1
print('assets',len(D),'rows',len(C),'per',[(s,len(x)) for s,x in D.items()]);
for dt in F.index:
 q=pd.concat([F.loc[dt],FR.loc[dt]],axis=1).dropna()
 if len(q)>=8: print('first',dt,len(q));break
print('coverage',F.notna().mean().mean())
