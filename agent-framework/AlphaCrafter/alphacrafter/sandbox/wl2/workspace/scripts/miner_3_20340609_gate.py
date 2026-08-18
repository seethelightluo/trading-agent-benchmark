import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   x=fn(s,days=6000)
   if x is not None and len(x)>300:return x
  except: pass
raw={s:fetch(s) for s in U}; p=pd.concat({s:x.set_index('date')['close'] for s,x in raw.items()},axis=1).sort_index(); r=np.log(p).diff(); loc=(p-p.rolling(120,min_periods=60).min())/(p.rolling(120,min_periods=60).max()-p.rolling(120,min_periods=60).min()+1e-12); down=r.where(r<0).rolling(20,min_periods=5).std(); total=r.rolling(60,min_periods=20).std(); f=((1-loc)/(1+down/(total+1e-12))).rank(axis=1,pct=True).shift(1)
for h in [1,10,20,40]:
 out=[]
 for d in f.index:
  a=pd.concat([f.loc[d],(p.shift(-h)/p-1).loc[d]],axis=1).dropna()
  if len(a)>=8: out.append((d,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
 q=pd.DataFrame(out,columns=['date','ic','n']).set_index('date').loc['2026-07-16':'2034-06-07']; print('GATE',h,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean(),q.n.mean())
PY