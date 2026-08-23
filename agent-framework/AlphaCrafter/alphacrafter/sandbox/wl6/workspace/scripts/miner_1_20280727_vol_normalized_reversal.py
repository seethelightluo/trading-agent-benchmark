import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   x=f(s,days=4000)
   if x is not None:return x
  except Exception: pass
R={}
for s in U:
 x=fetch(s)
 if x is not None:
  x=x.copy();x.date=pd.to_datetime(x.date);x=x.set_index('date').sort_index();R[s]=x.close.pct_change()
R=pd.DataFrame(R).sort_index(); n=len(R.columns)
# Signal available after t: recent reversal scaled by trailing volatility, with no future data.
vol=R.rolling(20,min_periods=10).std()
sig=-R.rolling(5,min_periods=5).sum()/vol
F={h:R.shift(-1).rolling(h).sum().shift(-(h-1)) for h in (1,5,10)}
for h in (1,5,10):
 rows=[]
 for dt in R.index:
  z=pd.DataFrame({'s':sig.loc[dt],'r':F[h].loc[dt]}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.s.nunique()>2: rows.append((dt,z.s.rank().corr(z.r.rank()),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']); m=q.ic.mean(); sd=q.ic.std(ddof=1)
 print('horizon',h,'dates',len(q),'avg_n',round(q.n.mean(),2),'IC',round(m,6),'ICIR',round(m/sd*np.sqrt(252),4),'hit',round((q.ic>0).mean(),4))
 for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31')]:
  v=q[(q.date.astype(str)>=a)&(q.date.astype(str)<=b)].ic; print('regime',a,'n',len(v),'IC',round(v.mean(),6) if len(v) else None)
print('assets',n,'coverage',round((sig.notna().mean().mean()),4),'dates',len(R))
