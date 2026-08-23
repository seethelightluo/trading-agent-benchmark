import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   d=f(s,days=4000)
   if d is not None and len(d): return d
  except Exception: pass
P={}
for s in U:
 d=fetch(s)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); d=d.set_index('date').sort_index(); P[s]=pd.to_numeric(d.close,errors='coerce').pct_change()
R=pd.DataFrame(P).sort_index(); m=R.mean(axis=1); rows=[]
for dt in R.index:
 sigs={}; fw={h:{} for h in [1,5,10]}
 for s in R.columns:
  r=R[s]; sig=(r.rolling(30,min_periods=20).sum()-m.rolling(30,min_periods=20).sum())/(r.rolling(20,min_periods=15).std()*np.sqrt(252)+.10)
  if dt not in sig.index or not np.isfinite(sig.loc[dt]): continue
  sigs[s]=sig.loc[dt]
  for h in fw: fw[h][s]=((1+r).shift(-1).rolling(h).apply(np.prod,raw=True).shift(-(h-1))-1).loc[dt]
 for h in fw:
  z=pd.DataFrame({'sig':pd.Series(sigs),'f':pd.Series(fw[h])}).dropna()
  if len(z)>=8: rows.append((dt,h,z.sig.rank().corr(z.f.rank()),len(z)))
q=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 x=q[q.h==h].dropna(); m=x.ic.mean(); sd=x.ic.std(ddof=1)
 print(f'{h}d dates={len(x)} avg_n={x.n.mean():.2f} coverage={x.n.mean()/15:.4f} IC={m:.6f} ICIR={m/sd*np.sqrt(252):.4f} hit={(x.ic>0).mean():.4f} turnover_proxy={x.ic.diff().abs().mean():.6f}')
 for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31'),('2029','2029-12-31')]:
  y=x[(x.date.astype(str)>=a)&(x.date.astype(str)<=b)].ic; print('regime',a,len(y),f'{y.mean():.6f}')
