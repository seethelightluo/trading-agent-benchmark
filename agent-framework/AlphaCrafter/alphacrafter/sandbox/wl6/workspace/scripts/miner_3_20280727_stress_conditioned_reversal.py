import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   d=f(s,days=4000)
   if d is not None:return d
  except Exception: pass
raw={}
for s in U:
 d=fetch(s)
 if d is None: continue
 d=d.copy();d.date=pd.to_datetime(d.date);d=d.set_index('date').sort_index()
 raw[s]=d.close.pct_change()
R=pd.DataFrame(raw).sort_index()
# Daily cross-asset dispersion, with strictly lagged rolling threshold.
disp=R.mad(axis=1) if hasattr(R,'mad') else R.sub(R.mean(axis=1),axis=0).abs().mean(axis=1)
th=disp.shift(1).rolling(60,min_periods=30).quantile(.75)
stress=(disp.shift(1)>th)
# signal known at close t, forward returns start t+1
sig=(-R.rolling(3).sum()).mul(stress.astype(float),axis=0)
F={h:R.shift(-1).rolling(h).sum().shift(-(h-1)) for h in [1,5,10]}
# equivalent forward compounded approximation for multi-day horizon
F[5]=R.shift(-1).rolling(5).sum().shift(-4);F[10]=R.shift(-1).rolling(10).sum().shift(-9)
for h in [1,5,10]:
 rows=[]
 for dt in R.index:
  z=pd.DataFrame({'s':sig.loc[dt],'r':F[h].loc[dt]}).dropna()
  if len(z)>=8 and z.s.nunique()>=3: rows.append((dt,z.s.rank().corr(z.r.rank()),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']);m=q.ic.mean();sd=q.ic.std(ddof=1)
 print('horizon',h,'dates',len(q),'avg_n',round(q.n.mean(),2),'IC',round(m,6),'ICIR',round(m/sd*np.sqrt(252),4),'hit',round((q.ic>0).mean(),4),'stress_days',int(stress.sum()))
 for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31')]:
  v=q[(q.date.astype(str)>=a)&(q.date.astype(str)<=b)].ic;print('regime',a,'n',len(v),'IC',round(v.mean(),6) if len(v) else None)
print('assets',len(raw),'coverage',round(R.notna().mean().mean(),4))
