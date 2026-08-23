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
 d=d.copy(); d.date=pd.to_datetime(d.date); d=d.set_index('date').sort_index(); c=d.close.astype(float); r=c.pct_change()
 mom=c.pct_change(20); vol=r.rolling(60).std()*np.sqrt(252)
 # Fraction of positive daily returns over the recent 20 sessions, centered at zero.
 persist=2*r.gt(0).rolling(20).mean()-1
 S[s]=pd.DataFrame({'mom':mom,'vol':vol,'persist':persist,'f1':c.shift(-1)/c-1,'f5':c.shift(-5)/c-1,'f10':c.shift(-10)/c-1})
D=sorted(set().union(*[x.index for x in S.values()])); rows=[]
for dt in D:
 a=[]
 for s,x in S.items():
  if dt in x.index and np.isfinite(x.loc[dt,['mom','vol','persist','f10']]).all():
   a.append((s,x.loc[dt,'mom'],x.loc[dt,'vol'],x.loc[dt,'persist'],x.loc[dt,'f1'],x.loc[dt,'f5'],x.loc[dt,'f10']))
 if len(a)>=8:
  z=pd.DataFrame(a,columns=['s','mom','vol','persist','f1','f5','f10']); med=z.mom.median()
  # Peer-relative momentum, penalized for unstable directional path and scaled by total risk.
  z['sig']=(z.mom-med)/(z.vol.replace(0,np.nan))*(0.5+0.5*z.persist)
  for h in [1,5,10]:
   q=z.dropna(subset=['sig',f'f{h}'])
   if len(q)>=8: rows.append((dt,h,q.sig.rank().corr(q[f'f{h}'].rank()),len(q)))
q=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 v=q[q.h==h].copy(); m=v.ic.mean(); sd=v.ic.std(ddof=1)
 print(f'{h}d dates={len(v)} avg_n={v.n.mean():.2f} coverage={v.n.mean()/15:.4f} IC={m:.6f} ICIR={m/sd*np.sqrt(252):.4f} hit={(v.ic>0).mean():.4f}')
 for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31'),('2029','2029-12-31')]:
  x=v[(v.date.astype(str)>=a)&(v.date.astype(str)<=b)].ic; print('regime',a,len(x),f'{x.mean():.6f}')
