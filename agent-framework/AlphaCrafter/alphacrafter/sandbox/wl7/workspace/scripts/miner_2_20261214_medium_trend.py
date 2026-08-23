import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data,get_account_dict
U=get_account_dict()['watch_list']; cutoff=pd.Timestamp('2026-12-13'); ds={}
for s in U:
 try: ds[s]=get_stock_daily_data(s,2600)
 except: ds[s]=get_index_daily_data(s,2600)
cl=pd.DataFrame({s:d.set_index('date')['close'] for s,d in ds.items()}).sort_index().loc[:cutoff]; r=cl.pct_change()
# medium trend, lagged, volatility adjusted
sig=(cl.shift(1)/cl.shift(21)-1)/(r.shift(1).rolling(60).std()*np.sqrt(60));
for h in [1,5,10]:
 y=cl.shift(-h)/cl-1; a=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]))
 a=np.array(a); print(h,len(a),np.nanmean(a),np.nanmean(a)/np.nanstd(a),np.mean(a>0))
print('coverage',sig.notna().sum(axis=1).mean()/15)
