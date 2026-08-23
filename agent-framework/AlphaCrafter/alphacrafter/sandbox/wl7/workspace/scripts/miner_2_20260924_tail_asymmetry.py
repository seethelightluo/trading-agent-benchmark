import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def f(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   x=fn(s,2400)
   if x is not None and len(x)>100:return x
  except: pass
D={}
for s in U:
 x=f(s)
 if x is not None:x=x.assign(date=pd.to_datetime(x.date).dt.normalize()).drop_duplicates('date').set_index('date').sort_index();D[s]=x
F={};R={}
for s,d in D.items():
 r=d.close.astype(float).pct_change(); mu=r.rolling(20,min_periods=15).mean(); sd=r.rolling(20,min_periods=15).std(); skew=((r-mu)**3).rolling(20,min_periods=15).mean()/sd**3
 F[s]=(-skew).shift(1);R[s]=r
A=sorted(set().union(*[set(x.index) for x in F.values()]));I=[];N=[];T=[];P=None;G={}
def S(a):
 a=np.array(a,float);a=a[np.isfinite(a)];return len(a),np.mean(a),np.mean(a)/np.std(a,ddof=1),np.mean(a>0)
for dt in A:
 z=[];vals={}
 for s in F:
  if dt in F[s].index and np.isfinite(F[s].loc[dt]):
   p=R[s].index.get_loc(dt)
   if p+1<len(R[s]):vals[s]=F[s].loc[dt];z.append((F[s].loc[dt],R[s].iloc[p+1]))
 if len(z)>=8:
  q=pd.Series([x for x,y in z]).corr(pd.Series([y for x,y in z]));I.append(q);N.append(len(z));G.setdefault(dt.year,[]).append(q)
  rr=pd.Series(vals).rank(pct=True)
  if P is not None:T.append(np.mean([abs(rr[k]-P[k]) for k in set(rr.index).intersection(set(P.index))]))
  P=rr
print('dates',len(I),'avg_names',np.mean(N),'assets',len(D),'total_dates',len(A),'coverage',len(I)/len(A),'turnover',np.mean(T));print('daily',S(I));print('regimes',{y:S(v) for y,v in G.items()})
