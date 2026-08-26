import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for g in (get_index_daily_data,get_stock_daily_data):
  try:
   x=g(s,days=4000)
   if x is not None and len(x): return x
  except: pass
D={s:fetch(s) for s in U};P=pd.DataFrame({s:x.set_index('date').close for s,x in D.items() if x is not None}).sort_index().ffill();R=P.pct_change(); disp=R.std(axis=1); gate=disp>disp.rolling(60).quantile(.75);F=(-R).where(gate,np.nan); FW=P.shift(-5)/P-1
rows=[]
for d in P.index:
 z=pd.concat([F.loc[d],FW.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
r=pd.DataFrame(rows,columns=['d','ic','n']).set_index('d');q=r.ic.dropna();print('dates',len(q),'mean_n',r.n.mean(),'coverage',r.n.sum()/(len(q)*15),'IC',q.mean(),'ICIR',q.mean()/q.std(),'hit',(q>0).mean())
for a,b in [('2020','2023-12-31'),('2024','2026-12-31'),('2027','2028-12-31'),('2029','2029-12-31'),('2028-04-05','2029-04-18')]:
 x=r.loc[a:b].ic.dropna(); print(a,len(x),x.mean(),x.mean()/x.std())
