import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,3000); d=d.copy(); d.date=pd.to_datetime(d.date); D[s]=d.sort_values('date').set_index('date').close
cl=pd.DataFrame(D).sort_index(); r=cl.pct_change();
# lagged 5d return divided by lagged 20d realized vol, residualized to cross-sectional median
f=(cl.pct_change(5)/r.rolling(20).std()).shift(1)
f=f.sub(f.median(axis=1),axis=0)
print('data',cl.shape,cl.index.min(),cl.index.max(),'coverage',f.notna().mean().mean())
for h in [1,5,10]:
 y=cl.shift(-h)/cl-1; arr=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: arr.append(z.iloc[:,0].corr(z.iloc[:,1]));ns.append(len(z))
 a=np.array(arr);print(h,len(a),np.mean(ns) if len(ns) else 0,a.mean() if len(a) else np.nan,a.mean()/a.std() if len(a) else np.nan,np.mean(a>0) if len(a) else np.nan)
f.to_csv('scripts/miner_3_20261217_riskadjusted5_signal.csv',index_label='date')
