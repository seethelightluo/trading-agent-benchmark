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
 dn=r.where(r<0,0).rolling(30,min_periods=20).std()
 S[s]=pd.DataFrame({'r20':c.pct_change(20),'dn':dn,'f1':c.shift(-1)/c-1,'f5':c.shift(-5)/c-1,'f10':c.shift(-10)/c-1})
D=sorted(set().union(*[x.index for x in S.values()])); rows=[]
for dt in D:
 a=[]
 for s,x in S.items():
  if dt in x.index and np.isfinite(x.loc[dt,['r20','dn','f1','f5','f10']]).all(): a.append((s,*x.loc[dt,['r20','dn','f1','f5','f10']].values))
 if len(a)>=8:
  z=pd.DataFrame(a,columns=['s','r20','dn','f1','f5','f10'])
  # Relative return removes common market direction; downside-risk denominator rewards resilient trends.
  z['sig']=(z.r20-z.r20.median())/(z.dn.clip(lower=0.002)*np.sqrt(30)+0.01)
  for h in [1,5,10]:
   q=z.dropna(subset=['sig',f'f{h}'])
   if len(q)>=8: rows.append((dt,h,q.sig.rank().corr(q[f'f{h}'].rank()),len(q)))
q=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 x=q[q.h==h].copy(); m=x.ic.mean(); sd=x.ic.std(ddof=1)
 print(f'{h}d dates={len(x)} avg_n={x.n.mean():.2f} coverage={x.n.mean()/15:.4f} IC={m:.6f} ICIR={m/sd*np.sqrt(252):.4f} hit={(x.ic>0).mean():.4f} ic_change={x.ic.diff().abs().mean():.6f}')
 for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31'),('2029','2029-12-31')]:
  y=x[(x.date.astype(str)>=a)&(x.date.astype(str)<=b)].ic; print('regime',a,len(y),f'{y.mean():.6f}')
