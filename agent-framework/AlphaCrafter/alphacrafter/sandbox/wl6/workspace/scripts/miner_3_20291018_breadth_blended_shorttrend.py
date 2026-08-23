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
 d=d.copy(); d.date=pd.to_datetime(d.date); d=d.set_index('date').sort_index(); c=pd.to_numeric(d.close,errors='coerce')
 S[s]=pd.DataFrame({'mom':c.pct_change(20),'short':c.pct_change(5),'above':(c>c.rolling(60,min_periods=40).mean()).astype(float),'f10':c.shift(-10)/c-1})
rows=[]
for dt in sorted(set().union(*[x.index for x in S.values()])):
 a=[]
 for s,x in S.items():
  if dt in x.index and np.isfinite(x.loc[dt,['mom','short','f10']]).all() and pd.notna(x.loc[dt,'above']): a.append((s,*x.loc[dt].values))
 if len(a)>=8:
  z=pd.DataFrame(a,columns=['s','mom','short','above','f10']); b=z.above.mean()
  # Momentum plus a conditional short-term component; coefficient changes by breadth.
  z['sig']=z.mom + (b-.5)*z.short
  q=z[['sig','f10']].dropna(); rows.append((dt,q.sig.rank().corr(q.f10.rank()),len(q),b))
q=pd.DataFrame(rows,columns=['date','ic','n','breadth']); m=q.ic.mean(); sd=q.ic.std(ddof=1)
print(f'10d dates={len(q)} avg_n={q.n.mean():.2f} coverage={q.n.mean()/15:.4f} IC={m:.6f} ICIR={m/sd*np.sqrt(252):.4f} hit={(q.ic>0).mean():.4f} turnover_proxy={q.ic.diff().abs().mean():.6f}')
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31'),('2029','2029-12-31')]:
 y=q[(q.date.astype(str)>=a)&(q.date.astype(str)<=b)].ic; print('regime',a,len(y),f'{y.mean():.6f}')
print('breadth weak/strong',q[q.breadth<.5].ic.mean(),q[q.breadth>=.5].ic.mean())
