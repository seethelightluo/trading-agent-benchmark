import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').sort_index().close.astype(float)
px=pd.concat(D,axis=1).sort_index(); r=px.pct_change(); vol=r.shift(1).rolling(20,min_periods=15).std()*np.sqrt(20)
# compare lookbacks, all predict next day, lagged and volatility scaled
rows=[]
for lb in [3,5,7,10,15,20]:
 f=-(px.shift(1)/px.shift(1+lb)-1)/vol
 for i,dt in enumerate(px.index[:-1]):
  z=pd.concat([f.loc[dt],px.iloc[i+1]/px.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: rows.append((lb,dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
o=pd.DataFrame(rows,columns=['lb','date','ic','n']); print('assets',px.shape[1],'dates',o.date.nunique())
for lb,z in o.groupby('lb'):
 q=z.ic; print('lb',lb,'dates',len(q),'avgN %.2f IC %.6f ICIR %.6f hit %.3f'%(z.n.mean(),q.mean(),q.mean()/q.std(),(q>0).mean()))
 s=(-(px.shift(1)/px.shift(1+lb)-1)/vol).stack().rename('signal').reset_index(); s.columns=['date','symbol','signal']; s.to_csv('scripts/miner_2_20280713_longrev%d_signal.csv'%lb,index=False)
