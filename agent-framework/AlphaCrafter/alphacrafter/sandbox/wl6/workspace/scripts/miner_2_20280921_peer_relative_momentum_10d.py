import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def f(s):
 for g in (get_index_daily_data,get_stock_daily_data):
  try:
   x=g(s,days=4000)
   if x is not None:return x
  except:pass
R={}
for s in U:
 x=f(s)
 if x is not None:
  x=x.set_index(pd.to_datetime(x.date)).sort_index(); z=np.log(x.close.astype(float)).diff(); R[s]=pd.DataFrame({'r':z,'m':z.rolling(10).sum(),'v':z.rolling(20,min_periods=15).std()})
M=pd.DataFrame({s:x.m for s,x in R.items()});V=pd.DataFrame({s:x.v for s,x in R.items()}); F0=pd.DataFrame({s:x.r for s,x in R.items()}); S=M.sub(M.median(axis=1),axis=0).div(V)
for h in [1,5,10]:
 F=F0.shift(-1) if h==1 else F0.shift(-1).rolling(h).sum().shift(-(h-1)); q=[]
 for d in S.index:
  z=pd.DataFrame({'s':S.loc[d],'r':F.loc[d]}).dropna()
  if len(z)>=8:q.append((d,z.s.rank().corr(z.r.rank()),len(z)))
 q=pd.DataFrame(q,columns=['d','ic','n']).dropna(); ic=q.ic; print(h,len(q),round(q.n.mean(),2),round(ic.mean(),6),round(ic.mean()/ic.std(ddof=1)*np.sqrt(252),4),round((ic>0).mean(),4))
 if h==1: print([(a,len(q[(q.d>=b)&(q.d<=c)]),round(q[(q.d>=b)&(q.d<=c)].ic.mean(),6)) for a,b,c in [('20-22','2020','2022-12-31'),('23-24','2023','2024-12-31'),('25-26','2025','2026-12-31'),('27-28','2027','2028-12-31')]])
