import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   d=f(s,days=4000)
   if d is not None:return d
  except (FileNotFoundError,KeyError,ValueError): pass
P={}
for s in U:
 d=fetch(s)
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); P[s]=d.set_index('date').sort_index().close
px=pd.DataFrame(P).sort_index(); ret=px.pct_change(); mkt=ret.mean(axis=1)
# Residual momentum: 20d asset return less rolling beta to the equal-weight benchmark times benchmark return.
beta=ret.rolling(60,min_periods=30).cov(mkt).div(mkt.rolling(60,min_periods=30).var(),axis=0)
res=ret.rolling(20).sum()-beta.mul(mkt.rolling(20).sum(),axis=0)
for h in [1,5,10]:
 fwd=px.shift(-h).div(px)-1; rows=[]
 for dt in px.index:
  z=pd.DataFrame({'s':res.loc[dt], 'r':fwd.loc[dt]}).dropna()
  if len(z)>=8: rows.append((dt,z.s.rank().corr(z.r.rank()),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']); mu=q.ic.mean(); sd=q.ic.std(ddof=1)
 print('h',h,'dates',len(q),'avg_n',round(q.n.mean(),2),'IC',round(mu,6),'ICIR',round(mu/sd*np.sqrt(252),4),'hit',round((q.ic>0).mean(),4),'coverage',round(q.n.mean()/15,4))
 for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31')]:
  v=q[(q.date.astype(str)>=a)&(q.date.astype(str)<=b)].ic; print(a,len(v),round(v.mean(),6) if len(v) else None)
