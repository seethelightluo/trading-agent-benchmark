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
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); d=d.set_index('date').sort_index()
  P[s]=pd.to_numeric(d.close,errors='coerce').pct_change()
R=pd.concat(P,axis=1).sort_index(); m=R.median(axis=1)
# residual trend: rolling beta to contemporaneous equal-weight benchmark, then residual cumulative return
out={}
for s in R:
 r=R[s]; beta=r.rolling(60,min_periods=40).cov(m)/(m.rolling(60,min_periods=40).var()+1e-8)
 resid=r-beta*m
 trend=resid.rolling(20,min_periods=15).sum()
 rv=resid.rolling(40,min_periods=25).std()
 out[s]=pd.DataFrame({'sig':trend/(rv*np.sqrt(40)+.03),'f1':R[s].shift(-1).rolling(1).sum().shift(-0),'f5':R[s].rolling(5).sum().shift(-5),'f10':R[s].rolling(10).sum().shift(-10),'f20':R[s].rolling(20).sum().shift(-20)})
# Correct forward returns directly from price-like cumulative log approximation
for s in R:
 for h in [1,5,10,20]: out[s][f'f{h}']=R[s].rolling(h).sum().shift(-h)
rows=[]
for dt in R.index:
 a=[]
 for s,x in out.items():
  if dt in x.index and np.isfinite(x.loc[dt]).all(): a.append((s,*x.loc[dt].values))
 if len(a)>=8:
  z=pd.DataFrame(a,columns=['s','sig','f1','f5','f10','f20'])
  for h in [1,5,10,20]:
   v=z[['sig',f'f{h}']].dropna(); rows.append((dt,h,v.sig.rank().corr(v[f'f{h}'].rank()),len(v)))
q=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10,20]:
 x=q[q.h==h]; mu=x.ic.mean(); sd=x.ic.std(ddof=1)
 print(f'{h}d dates={len(x)} avg_n={x.n.mean():.2f} coverage={x.n.mean()/15:.4f} IC={mu:.6f} ICIR={mu/sd*np.sqrt(252):.4f} hit={(x.ic>0).mean():.4f} turnover_proxy={x.ic.diff().abs().mean():.6f}')
 for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31'),('2029','2029-12-31')]:
  y=x[(x.date.astype(str)>=a)&(x.date.astype(str)<=b)].ic; print('regime',a,len(y),f'{y.mean():.6f}')
