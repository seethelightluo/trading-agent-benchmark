import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   d=f(s,days=4000)
   if d is not None and len(d): return d
  except Exception: pass
C={}
for s in U:
 d=fetch(s)
 if d is not None:
  x=d.copy(); x.date=pd.to_datetime(x.date); C[s]=pd.to_numeric(x.set_index('date').sort_index().close,errors='coerce')
P=pd.DataFrame(C).sort_index(); r=P.pct_change(); fwd=P.shift(-10)/P-1
v20=r.rolling(20,min_periods=15).std(); v120=r.rolling(120,min_periods=60).std()
# Contrarian short-horizon return, amplified only when current volatility is elevated vs its own long baseline.
reg=(v20/(v120+1e-9)-1).clip(-.5,1.0)
sig=-P.pct_change(10)/(v20*np.sqrt(10)+.01)*(1+.6*reg)
rows=[]
for dt in P.index:
 z=pd.DataFrame({'s':sig.loc[dt],'f':fwd.loc[dt]}).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8: rows.append((dt,z.s.rank().corr(z.f.rank()),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']); m=q.ic.mean(); sd=q.ic.std(ddof=1)
print('idea regime_gated_reversal dates',len(q),'avg_n',q.n.mean(),'coverage',q.n.mean()/15,'IC %.6f ICIR %.4f hit %.4f turnover %.6f'%(m,m/sd*np.sqrt(252),(q.ic>0).mean(),q.ic.diff().abs().mean()))
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31'),('2029','2029-12-31')]:
 y=q[(q.date.astype(str)>=a)&(q.date.astype(str)<=b)].ic; print(a,len(y),round(y.mean(),6) if len(y) else None)
