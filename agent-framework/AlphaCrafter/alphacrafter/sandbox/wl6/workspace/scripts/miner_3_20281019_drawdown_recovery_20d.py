import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    for f in (get_index_daily_data,get_stock_daily_data):
        try:
            d=f(s,days=4000)
            if d is not None:return d
        except Exception: pass
S={}
for s in U:
 d=fetch(s)
 if d is None: continue
 d=d.copy(); d.date=pd.to_datetime(d.date); d=d.set_index('date').sort_index(); c=d.close; r=c.pct_change()
 # Recovery signal: medium-term risk-adjusted return plus normalized rebound from the trailing 60d trough.
 vol=r.rolling(40).std(); mom=r.rolling(20).sum()/(vol*np.sqrt(20))
 trough=c.rolling(60).min(); recovery=(c/trough-1).clip(0,2)
 sig=mom + 0.50*recovery
 S[s]=pd.DataFrame({'sig':sig,'f1':c.pct_change().shift(-1),'f5':c.shift(-5)/c-1,'f10':c.shift(-10)/c-1})
rows=[]
for dt in sorted(set().union(*[x.index for x in S.values()])):
 a=[(x.loc[dt].sig,x.loc[dt].f10) for x in S.values() if dt in x.index and np.isfinite(x.loc[dt].sig) and np.isfinite(x.loc[dt].f10)]
 if len(a)>=8:
  z=pd.DataFrame(a,columns=['s','r']); rows.append((dt,z.s.rank().corr(z.r.rank()),len(a)))
q=pd.DataFrame(rows,columns=['date','ic','n']); m=q.ic.mean(); sd=q.ic.std(ddof=1)
print('10d dates',len(q),'avg_n',round(q.n.mean(),2),'IC',round(m,6),'ICIR',round(m/sd*np.sqrt(252),4),'hit',round((q.ic>0).mean(),4),'coverage',round(q.n.mean()/15,4),'assets',len(S))
for h,shift in [('1d',1),('5d',5),('10d',10)]:
 rows=[]
 for dt in sorted(set().union(*[x.index for x in S.values()])):
  a=[(x.loc[dt].sig,x.loc[dt][f'f{shift}']) for x in S.values() if dt in x.index and np.isfinite(x.loc[dt].sig) and np.isfinite(x.loc[dt][f'f{shift}'])]
  if len(a)>=8:
   z=pd.DataFrame(a,columns=['s','r']); rows.append(z.s.rank().corr(z.r.rank()))
 v=pd.Series(rows).dropna(); print(h,'dates',len(v),'IC',round(v.mean(),6),'ICIR',round(v.mean()/v.std(ddof=1)*np.sqrt(252),4),'hit',round((v>0).mean(),4))
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31')]:
 v=q[(q.date.astype(str)>=a)&(q.date.astype(str)<=b)].ic; print('regime',a,len(v),round(v.mean(),6) if len(v) else None)
