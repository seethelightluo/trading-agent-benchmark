import numpy as np,pandas as pd,json,glob
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for a in A:
 d=get_stock_daily_data(a,days=3000)
 if d is not None and len(d)>120:px[a]=d.set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().ffill();r=p.pct_change();v=r.rolling(20,min_periods=10).std()
base=-p.pct_change(3); cand=base.where(v.le(v.quantile(.75,axis=1),axis=0))
# compare candidate with library-like signals, flatten aligned valid pairs
signals={'candidate':cand,'3d':base,'5d':-p.pct_change(5),'riskadj':-p.pct_change(5)/v}
for k,x in signals.items():
 z=pd.concat([cand.stack().rename('c'),x.stack().rename(k)],axis=1).dropna(); print(k,len(z),z.corr().iloc[0,1])
print('candidate dates',cand.dropna(how='all').shape[0])
