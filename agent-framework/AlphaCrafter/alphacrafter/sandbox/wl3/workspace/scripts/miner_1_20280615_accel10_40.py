import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in [get_index_daily_data,get_stock_daily_data]:
  try:
   x=fn(s,days=4000)
   if x is not None and len(x)>100:return x[['date','close']]
  except: pass
D={s:get(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
P=pd.concat([x.set_index('date').close.rename(s) for s,x in D.items()],axis=1).sort_index().ffill(); L=np.log(P)
# acceleration: recent 10d return minus average daily return over preceding 40d, all information lagged one day
f=(L.shift(1)-L.shift(11))-((L.shift(1)-L.shift(41))/4)
rows=[]
for h in [1,3,5,10]:
 F=P.shift(-h)/P-1; qs=[]; ns=[]; ds=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],F.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   z=a.iloc[:,0].rank().corr(a.iloc[:,1].rank())
   if pd.notna(z):qs.append(z);ns.append(len(a));ds.append(dt)
 q=pd.Series(qs,index=pd.to_datetime(ds)); print(h,q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean(),len(q),np.mean(ns))
# 10d regime
q10=pd.Series(qs,index=pd.to_datetime(ds)) if h==10 else None
print('coverage',f.notna().mean().mean(),'turnover',(f.rank(axis=1,pct=True).diff().abs().mean(axis=1)>0.05).mean())
f.to_csv('scripts/miner_1_20280615_accel10_40_signal.csv')
