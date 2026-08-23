import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   d=f(s,days=4000)
   if d is not None:return d
  except (FileNotFoundError,KeyError,ValueError):pass
S={}
for s in U:
 d=fetch(s)
 if d is None:continue
 d=d.copy(); d.date=pd.to_datetime(d.date); d=d.set_index('date').sort_index()
 clv=(2*d.close-d.high-d.low)/(d.high-d.low).replace(0,np.nan)
 # Lagged completed-session signal: CLV persistence normalized by trailing realized risk.
 sig=clv.rolling(5).mean()/d.close.pct_change().rolling(20).std().replace(0,np.nan)
 S[s]=pd.DataFrame({'sig':sig,'f1':d.close.pct_change().shift(-1),'f5':d.close.shift(-5)/d.close-1,'f10':d.close.shift(-10)/d.close-1})
for h in ['f1','f5','f10']:
 rows=[]
 for dt in sorted(set().union(*[x.index for x in S.values()])):
  a=[(x.loc[dt].sig,x.loc[dt,h]) for x in S.values() if dt in x.index and np.isfinite(x.loc[dt].sig) and np.isfinite(x.loc[dt,h])]
  if len(a)>=8:
   z=pd.DataFrame(a,columns=['s','r']); rows.append((dt,z.s.rank().corr(z.r.rank()),len(a)))
 q=pd.DataFrame(rows,columns=['date','ic','n']); m=q.ic.mean(); sd=q.ic.std(ddof=1)
 print(h,'dates',len(q),'avg_n',round(q.n.mean(),2),'IC',round(m,6),'ICIR',round(m/sd*np.sqrt(252),4),'hit',round((q.ic>0).mean(),4))
 for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31')]:
  v=q[(q.date.astype(str)>=a)&(q.date.astype(str)<=b)].ic; print(a,len(v),round(v.mean(),6) if len(v) else None)
print('assets',len(S),'coverage',round(q.n.mean()/15,4))
