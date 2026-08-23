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
 r=d.close.pct_change()
 # relative momentum versus contemporaneous peer median, all inputs lagged one day
 mom=d.close.pct_change(20).shift(1)
 peer=pd.DataFrame({x: (R[x] if False else np.nan) for x in []})
 R[s]=pd.DataFrame({'r':r,'mom':mom,'vol':r.rolling(20).std().shift(1)})
M=pd.DataFrame({s:x.mom for s,x in R.items()}); V=pd.DataFrame({s:x.vol for s,x in R.items()}); F0=pd.DataFrame({s:x.r for s,x in R.items()})
peer=M.median(axis=1); sig=M.sub(peer,axis=0).div(V.replace(0,np.nan))
for h in [1,5,10]:
 if h==1: F=F0.shift(-1)
 else: F=F0.shift(-1).rolling(h).sum().shift(-(h-1))
 rows=[]
 for dt in M.index:
  z=pd.DataFrame({'s':sig.loc[dt],'r':F.loc[dt]}).dropna()
  if len(z)>=8 and z.s.nunique()>=3: rows.append((dt,z.s.rank().corr(z.r.rank()),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']);m=q.ic.mean();sd=q.ic.std(ddof=1)
 print('horizon',h,'dates',len(q),'avg_n',round(q.n.mean(),2),'IC',round(m,6),'ICIR',round(m/sd*np.sqrt(252),4),'hit',round((q.ic>0).mean(),4))
 for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31')]:
  v=q[(q.date.astype(str)>=a)&(q.date.astype(str)<=b)].ic; print('regime',a,'n',len(v),'IC',round(v.mean(),6) if len(v) else None)
print('assets',len(R),'coverage',round(q.n.mean()/15,4))
