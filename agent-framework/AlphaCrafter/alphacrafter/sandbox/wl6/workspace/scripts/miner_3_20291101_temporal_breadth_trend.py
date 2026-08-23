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
 if d is None: continue
 d=d.copy();d.date=pd.to_datetime(d.date);d=d.set_index('date').sort_index()
 C[s]=pd.to_numeric(d.close,errors='coerce')
P=pd.DataFrame(C).sort_index(); ma=P.rolling(60,min_periods=40).mean(); above=P>ma
breadth=above.mean(axis=1,skipna=True); sm=breadth.rolling(20,min_periods=10).mean()
ret=P.pct_change(20); f=P.shift(-10)/P-1
rows=[]
for dt in P.index:
 z=pd.DataFrame({'sig':ret.loc[dt]*(.5+sm.loc[dt]),'fw':f.loc[dt]}).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8: rows.append((dt,z.sig.rank().corr(z.fw.rank()),len(z),breadth.loc[dt]))
q=pd.DataFrame(rows,columns=['date','ic','n','breadth']);m=q.ic.mean();sd=q.ic.std(ddof=1)
print('instruments',len(P.columns),'dates',len(P),'observations',len(q),'avg_n',q.n.mean(),'coverage',q.n.mean()/15)
print('10d IC=%.6f ICIR=%.4f hit=%.4f turnover_proxy=%.6f'%(m,m/sd*np.sqrt(252),(q.ic>0).mean(),q.ic.diff().abs().mean()))
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31'),('2029','2029-12-31')]:
 y=q[(q.date.astype(str)>=a)&(q.date.astype(str)<=b)].ic; print('regime',a,'dates',len(y),'IC=%.6f'%y.mean() if len(y) else 'none')
print('breadth weak/strong',q[q.breadth<.5].ic.mean(),q[q.breadth>=.5].ic.mean())
