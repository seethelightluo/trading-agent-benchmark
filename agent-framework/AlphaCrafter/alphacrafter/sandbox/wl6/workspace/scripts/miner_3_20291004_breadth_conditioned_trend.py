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
 c=pd.to_numeric(d.close,errors='coerce')
 S[s]=pd.DataFrame({'mom':c.pct_change(20),'above60':c>c.rolling(60,min_periods=40).mean(),**{f'f{h}':c.shift(-h)/c-1 for h in [1,5,10,20]}})
rows=[]
for dt in sorted(set().union(*[x.index for x in S.values()])):
 a=[]
 for s,x in S.items():
  if dt in x.index and np.isfinite(x.loc[dt,['mom','f1','f5','f10','f20']]).all() and pd.notna(x.loc[dt,'above60']): a.append((s,*x.loc[dt].values))
 if len(a)>=8:
  z=pd.DataFrame(a,columns=['s','mom','above60','f1','f5','f10','f20']); breadth=z.above60.mean()
  z['sig']=z.mom if breadth>=0.5 else -z.mom
  for h in [1,5,10,20]:
   q=z[['sig',f'f{h}']].dropna(); rows.append((dt,h,q.sig.rank().corr(q[f'f{h}'].rank()),len(q),breadth))
q=pd.DataFrame(rows,columns=['date','h','ic','n','breadth'])
for h in [1,5,10,20]:
 x=q[q.h==h]; m=x.ic.mean(); sd=x.ic.std(ddof=1)
 print(f'{h}d dates={len(x)} avg_n={x.n.mean():.2f} coverage={x.n.mean()/15:.4f} IC={m:.6f} ICIR={m/sd*np.sqrt(252):.4f} hit={(x.ic>0).mean():.4f} turnover_proxy={x.ic.diff().abs().mean():.6f}')
 for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31'),('2029','2029-12-31')]:
  y=x[(x.date.astype(str)>=a)&(x.date.astype(str)<=b)].ic; print('regime',a,len(y),f'{y.mean():.6f}')
 print('breadth weak/strong',x[x.breadth<.5].ic.mean(),x[x.breadth>=.5].ic.mean())
