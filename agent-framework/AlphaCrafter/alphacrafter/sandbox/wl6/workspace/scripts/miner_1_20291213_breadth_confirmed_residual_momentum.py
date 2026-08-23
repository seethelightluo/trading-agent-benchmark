import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,days=4000)
   if d is not None and len(d): return d
  except Exception: pass
C={}
for s in U:
 d=fetch(s)
 if d is not None and len(d):
  x=d.copy(); x.date=pd.to_datetime(x.date)
  C[s]=pd.to_numeric(x.set_index('date').sort_index().close,errors='coerce')
P=pd.DataFrame(C); P.index=pd.to_datetime(P.index); P=P.sort_index(); r=P.pct_change(); fwd=P.shift(-10)/P-1
r20=P.pct_change(20); vol=r.rolling(30,min_periods=20).std()
# Cross-sectional residual momentum, confirmed by breadth: breadth is fraction of assets with positive 20d return.
med=r20.median(axis=1); breadth=(r20>0).mean(axis=1)
sig=r20.sub(med,axis=0)/(vol*np.sqrt(20)+.01) .mul(0.5 + breadth, axis=0)
rows=[]
for dt in P.index:
 z=pd.concat([sig.loc[dt].rename('s'),fwd.loc[dt].rename('f')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8: rows.append((dt,z.s.rank().corr(z.f.rank()),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']); m=q.ic.mean(); sd=q.ic.std(ddof=1)
print('idea breadth_confirmed_residual_momentum dates',len(q),'avg_n',q.n.mean(),'coverage',q.n.mean()/15,'IC %.6f ICIR %.4f hit %.4f turnover %.6f'%(m,m/sd*np.sqrt(252),(q.ic>0).mean(),q.ic.diff().abs().mean()))
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31'),('2029','2029-12-31')]:
 y=q[(q.date.astype(str)>=a)&(q.date.astype(str)<=b)].ic; print(a,len(y),round(y.mean(),6) if len(y) else None)
# horizon decay
for h in [5,10,20]:
 fw=P.shift(-h)/P-1; rr=[]
 for dt in P.index:
  z=pd.concat([sig.loc[dt].rename('s'),fw.loc[dt].rename('f')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8: rr.append(z.s.rank().corr(z.f.rank()))
 print('decay',h,len(rr),np.nanmean(rr),np.nanmean(rr)/np.nanstd(rr,ddof=1)*np.sqrt(252))
