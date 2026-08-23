import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   d=f(s,days=4000)
   if d is not None and len(d): return d
  except Exception: pass
S={}
for s in U:
 d=fetch(s)
 if d is None: continue
 d=d.copy(); d.date=pd.to_datetime(d.date); d=d.set_index('date').sort_index()
 c=pd.to_numeric(d.close,errors='coerce'); r=c.pct_change(); v=pd.to_numeric(d.volume,errors='coerce')
 vol=r.rolling(20,min_periods=15).std(); vs=(v/v.rolling(20,min_periods=10).median()).clip(0.25,4).fillna(1)
 # Contrarian 5d return, quality-scaled by volatility and modestly confirmed by abnormal volume.
 sig=-(c.pct_change(5)/(vol.clip(lower=.002)*np.sqrt(20)+.01))*(0.75+0.25*vs.rank(pct=True))
 S[s]=pd.DataFrame({'sig':sig,**{f'f{h}':c.shift(-h)/c-1 for h in [1,5,10]}})
D=sorted(set().union(*[x.index for x in S.values()])); rows=[]
for dt in D:
 a=[]
 for s,x in S.items():
  if dt in x.index and np.isfinite(x.loc[dt]).all(): a.append((s,*x.loc[dt].values))
 if len(a)>=8:
  z=pd.DataFrame(a,columns=['s','sig','f1','f5','f10'])
  for h in [1,5,10]:
   q=z.dropna(subset=['sig',f'f{h}']); rows.append((dt,h,q.sig.rank().corr(q[f'f{h}'].rank()),len(q)))
q=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 x=q[q.h==h].dropna(); m=x.ic.mean(); sd=x.ic.std(ddof=1)
 print(f'{h}d dates={len(x)} avg_n={x.n.mean():.2f} coverage={x.n.mean()/15:.4f} IC={m:.6f} ICIR={m/sd*np.sqrt(252):.4f} hit={(x.ic>0).mean():.4f}')
 for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31'),('2029','2029-12-31')]:
  y=x[(x.date.astype(str)>=a)&(x.date.astype(str)<=b)].ic; print('regime',a,len(y),f'{y.mean():.6f}')
