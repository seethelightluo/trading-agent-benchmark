import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,5000)
   if x is not None and len(x)>=100:return x
  except Exception: pass
D={s:load(s) for s in U}; C=pd.DataFrame({s:x.set_index(pd.to_datetime(x.date)).close.astype(float) for s,x in D.items() if x is not None}).sort_index(); R=C.pct_change(); res=R.sub(R.mean(axis=1),axis=0)
for w in [2,3,5]:
 f=-res.rolling(w).sum().shift(1)/R.rolling(20).std().shift(1).replace(0,np.nan); vals=[]
 for d in f.index:
  q=pd.concat([f.loc[d],R.shift(-1).loc[d]],axis=1).dropna()
  if len(q)>=8: vals.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 v=pd.Series(vals).dropna(); q=v.tail(120); print('W',w,'dates',len(v),'IC %.6f ICIR %.6f recent %.6f recentIR %.6f'%(v.mean(),v.mean()/v.std(),q.mean(),q.mean()/q.std()))
