import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for s in U:
 d=get_stock_daily_data(s,days=2100)
 if d is not None: raw[s]=d.sort_values('date').set_index('date').close.astype(float)
bench=raw['SPX']; rows={}
for s,c in raw.items():
 z=pd.concat([c.rename('c'),bench.rename('b')],axis=1).dropna(); r=z.pct_change();
 beta=r['c'].rolling(60,min_periods=40).cov(r['b'])/r['b'].rolling(60,min_periods=40).var()
 resid=r['c']-beta*r['b']; f=resid.rolling(20,min_periods=15).sum()
 y=c.shift(-1)/c-1
 rows[s]=pd.concat([f.rename('f'),y.rename('y')],axis=1)
ics=[]; ns=[]; turns=[]; prev=None; dates=[]
for dt in sorted(set().union(*[set(x.index) for x in rows.values()])):
 v=[];y=[];names=[]
 for s,z in rows.items():
  if dt in z.index and np.isfinite(z.loc[dt]).all():v.append(z.loc[dt,'f']);y.append(z.loc[dt,'y']);names.append(s)
 if len(v)>=8:
  q=pd.Series(v).corr(pd.Series(y),method='spearman')
  if np.isfinite(q):
   ics.append(q);ns.append(len(v));dates.append(dt)
   rk=dict(zip(names,pd.Series(v).rank(pct=True)))
   if prev: turns.append(np.mean([abs(rk[s]-prev[s]) for s in set(rk)&set(prev)]))
   prev=rk
x=np.array(ics); print('factor=60d market-residual 20d momentum; dates',len(x),'avg_names',np.mean(ns),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',np.mean(x>0),'turnover',np.mean(turns),'coverage',np.mean(ns)/15)
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2027')]:
 q=x[[a<=str(d)[:4]<=b for d in dates]]
 if len(q)>1: print(a,b,len(q),q.mean(),q.mean()/q.std(ddof=1))
