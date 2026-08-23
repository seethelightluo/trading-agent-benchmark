import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   d=f(s,days=4000)
   if d is not None:return d
  except Exception: pass
R={}
for s in U:
 d=fetch(s)
 if d is None: continue
 d=d.copy();d.date=pd.to_datetime(d.date);d=d.set_index('date').sort_index()
 r=np.log(d.close.astype(float)).diff()
 R[s]=pd.DataFrame({'r':r,'mom':r.rolling(5,min_periods=5).sum(),'vol':r.rolling(20,min_periods=15).std()})
M=pd.DataFrame({s:x.mom for s,x in R.items()}); V=pd.DataFrame({s:x.vol for s,x in R.items()}); F0=pd.DataFrame({s:x.r for s,x in R.items()})
sig=M.sub(M.median(axis=1),axis=0).div(V.replace(0,np.nan)); rank=sig.rank(axis=1,pct=True)
for h in [1,5,10]:
 F=F0.shift(-1) if h==1 else F0.shift(-1).rolling(h).sum().shift(-(h-1))
 rows=[]
 for dt in sig.index:
  z=pd.DataFrame({'s':sig.loc[dt],'r':F.loc[dt]}).dropna()
  if len(z)>=8 and z.s.nunique()>=3: rows.append((dt,z.s.rank().corr(z.r.rank()),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']); q=q[np.isfinite(q.ic)]
 m=q.ic.mean();sd=q.ic.std(ddof=1)
 print('horizon',h,'dates',len(q),'avg_n',round(q.n.mean(),2),'coverage',round(q.n.mean()/len(U),4),'IC',round(m,6),'ICIR',round(m/sd*np.sqrt(252),4),'hit',round((q.ic>0).mean(),4))
 if h==1:
  print('regimes',[(a,len(q[(q.date>=b)&(q.date<=c)]),round(q[(q.date>=b)&(q.date<=c)].ic.mean(),6)) for a,b,c in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027-28','2027','2028-12-31')]])
print('instruments',len(R),'dates',len(M))
