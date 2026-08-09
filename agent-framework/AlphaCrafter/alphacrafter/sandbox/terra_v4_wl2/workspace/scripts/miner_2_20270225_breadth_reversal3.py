import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None:return x
  except Exception: pass
px=pd.DataFrame({s:g(s).set_index('date')['close'] for s in U}).sort_index(); r=px.pct_change(); fr=px.shift(-1)/px-1
# prior-day breadth determines whether 3d reversal is active; neutral cross-sectional centering
for t in [.60,.70,.80,.90]:
 breadth=r.lt(0).sum(axis=1).div(r.notna().sum(axis=1)).shift(1)
 f=(-r.rolling(3).sum()).where(breadth>=t); f=f.sub(f.median(axis=1),axis=0)
 vals=[]; ns=[]; dates=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); dates.append(dt)
 a=np.array(vals); print(t,len(a),round(np.mean(ns),2),round(np.mean(a),6),round(np.mean(a)/np.std(a,ddof=1),6),round(np.mean(a>0),4),round(f.notna().sum().sum()/len(U)/len(f),4),dates[0],dates[-1])
# artifact for best broad threshold .6
breadth=r.lt(0).sum(axis=1).div(r.notna().sum(axis=1)).shift(1); f=(-r.rolling(3).sum()).where(breadth>=.60); f=f.sub(f.median(axis=1),axis=0)
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('../persistent/factor_signals_miner_2_20270225_breadth_reversal3.csv',index=False)
print('artifact',len(out))
