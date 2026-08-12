import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,5000)
   if x is not None and len(x)>=100:return x
  except Exception: pass
D={s:load(s) for s in U}; C=pd.DataFrame({s:x.set_index(pd.to_datetime(x.date)).close.astype(float) for s,x in D.items() if x is not None}).sort_index(); R=C.pct_change()
try:
 v=get_index_daily_data('VIX',5000); V=v.set_index(pd.to_datetime(v.date)).close.astype(float).reindex(C.index).ffill()
except Exception: V=pd.Series(index=C.index,dtype=float)
res=R.sub(R.mean(axis=1),axis=0); vol=R.rolling(20).std().shift(1)
vcut=V.rolling(252,min_periods=100).quantile(.60).shift(1)
for th,pers in [(.60,3),(.55,3),(.60,2)]:
 shock=res.rolling(5).sum().shift(1); breadth=(R.rolling(5).sum().shift(1)>0).mean(axis=1)
 active=((breadth<th)&(V>vcut)).astype(float).rolling(pers,min_periods=1).max()
 f=(-shock/vol).mul(active.replace(0,np.nan),axis=0); rows=[]
 for d in f.index:
  q=pd.concat([f.loc[d],R.shift(-1).loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1: rows.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 o=pd.Series(rows).dropna(); print('variant',th,pers,'assets',len(C.columns),'dates',len(o),'avg_n',round(f.notna().sum(axis=1).mean(),3),'IC %.6f ICIR %.6f hit %.4f'%(o.mean(),o.mean()/o.std(),(o>0).mean()))
 if th==.60 and pers==3:f.to_csv('scripts/miner_2_20321014_adaptive_stress_reversal_signal.csv',index_label='date')
# recent and decay for best candidate
q=pd.Series(rows).dropna(); print('recent120 ICIR',q.tail(120).mean()/q.tail(120).std())
for h in [3,5,10]:
 rr=C.pct_change(h).shift(-h); z=[]
 for d in f.index:
  a=pd.concat([f.loc[d],rr.loc[d]],axis=1).dropna()
  if len(a)>=8 and a.iloc[:,0].nunique()>1:z.append(a.iloc[:,0].rank().corr(a.iloc[:,1].rank()))
 print('decay',h,np.nanmean(z),len(z))
