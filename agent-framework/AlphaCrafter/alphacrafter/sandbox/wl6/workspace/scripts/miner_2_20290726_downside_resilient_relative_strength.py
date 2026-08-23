import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   d=f(s,days=4000)
   if d is not None and len(d): return d
  except Exception: pass
R={}
for s in U:
 d=fetch(s)
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); d=d.set_index('date').sort_index()
  R[s]=pd.to_numeric(d.close,errors='coerce').pct_change()
R=pd.concat(R,axis=1).sort_index()
# Downside-resilient relative strength: lagged 20d return relative to universe median,
# rewarded for positive-day consistency and penalized by downside semideviation.
ret20=R.rolling(20,min_periods=15).sum(); rel=ret20-ret20.median(axis=1).values[:,None]
down=R.clip(upper=0).pow(2).rolling(30,min_periods=20).mean().pow(.5)
pos=(R>0).rolling(30,min_periods=20).mean()
sig=rel/(down+0.01)*(0.5+pos)
rows=[]
for dt in R.index:
 for h in [1,5,10,20]:
  f=R.shift(-1).rolling(h).sum().shift(-(h-1)).loc[dt]
  q=pd.DataFrame({'s':sig.loc[dt],'f':f}).dropna()
  if len(q)>=8: rows.append((dt,h,q.s.rank().corr(q.f.rank()),len(q)))
q=pd.DataFrame(rows,columns=['date','h','ic','n'])
print('universe',R.shape[1],'rows',len(R),'range',R.index.min(),R.index.max())
for h in [1,5,10,20]:
 x=q[q.h==h]; mu=x.ic.mean(); sd=x.ic.std(ddof=1)
 print(f'{h}d dates={len(x)} avg_n={x.n.mean():.2f} coverage={x.n.mean()/15:.4f} IC={mu:.6f} ICIR={mu/sd*np.sqrt(252):.4f} hit={(x.ic>0).mean():.4f} turnover_proxy={x.ic.diff().abs().mean():.6f}')
 for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31'),('2029','2029-12-31')]:
  y=x[(x.date.astype(str)>=a)&(x.date.astype(str)<=b)].ic
  print('regime',a,len(y),f'{y.mean():.6f}')
