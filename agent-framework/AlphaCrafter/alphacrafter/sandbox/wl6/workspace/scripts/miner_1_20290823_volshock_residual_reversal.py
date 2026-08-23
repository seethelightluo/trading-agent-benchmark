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
 c=pd.to_numeric(d.close,errors='coerce'); r=c.pct_change()
 resid=r.rolling(10,min_periods=10).sum()-r.rolling(10,min_periods=10).mean() # cross-sectional residual formed below
 v30=r.rolling(30,min_periods=20).std()*np.sqrt(252)
 v20=r.rolling(20,min_periods=15).std(); v120=r.rolling(120,min_periods=80).std()
 shock=(v20/(v120+1e-8)).clip(0.5,2.0)
 S[s]=pd.DataFrame({'ret10':c/c.shift(10)-1,'v30':v30,'shock':shock,**{f'f{h}':c.shift(-h)/c-1 for h in [1,5,10,20]}})
rows=[]
for dt in sorted(set().union(*[x.index for x in S.values()])):
 a=[]
 for s,x in S.items():
  if dt in x.index:
   z=x.loc[dt]
   if np.isfinite(z[['ret10','v30','shock','f1','f5','f10','f20']]).all(): a.append((s,*z.values))
 if len(a)>=8:
  z=pd.DataFrame(a,columns=['s','ret10','v30','shock','f1','f5','f10','f20'])
  z['resid']=z.ret10-z.ret10.mean()
  # Asset-specific volatility-shock amplification changes ranks.
  z['sig']=(-z.resid/(z.v30+0.02))*(1+0.6*(z.shock-1).clip(-0.5,1.0))
  for h in [1,5,10,20]:
   q=z[['sig',f'f{h}']].dropna(); rows.append((dt,h,q.sig.rank().corr(q[f'f{h}'].rank()),len(q)))
q=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10,20]:
 x=q[q.h==h]; m=x.ic.mean(); sd=x.ic.std(ddof=1)
 print(f'{h}d dates={len(x)} avg_n={x.n.mean():.2f} coverage={x.n.mean()/15:.4f} IC={m:.6f} ICIR={m/sd*np.sqrt(252):.4f} hit={(x.ic>0).mean():.4f} turnover_proxy={x.ic.diff().abs().mean():.6f}')
 for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31'),('2029','2029-12-31')]:
  y=x[(x.date.astype(str)>=a)&(x.date.astype(str)<=b)].ic; print('regime',a,len(y),f'{y.mean():.6f}')
