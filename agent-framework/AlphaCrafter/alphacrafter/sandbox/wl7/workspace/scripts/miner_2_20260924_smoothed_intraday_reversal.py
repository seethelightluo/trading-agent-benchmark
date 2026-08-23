import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   x=fn(s,2400)
   if x is not None and len(x)>100:return x
  except Exception: pass
D={}
for s in U:
 x=fetch(s)
 if x is not None:
  x=x.copy();x.date=pd.to_datetime(x.date).dt.normalize();D[s]=x.drop_duplicates('date').set_index('date').sort_index()
F={};R={}
for s,d in D.items():
 o=d.open.astype(float);c=d.close.astype(float);r=c.pct_change()
 # smoothed three-session intraday reversal, lagged; reduces one-day signal noise
 intr=(1-c/o).rolling(3,min_periods=3).mean().shift(1)
 F[s]=intr;R[s]=r
all_dates=sorted(set().union(*[set(x.index) for x in F.values()]));ics=[];obs=[];turn=[];prev=None;reg={}
def stat(a):
 a=np.asarray(a,float);a=a[np.isfinite(a)];return len(a),float(np.mean(a)),float(np.mean(a)/np.std(a,ddof=1)),float(np.mean(a>0))
for dt in all_dates:
 vals={s:F[s].loc[dt] for s in F if dt in F[s].index and np.isfinite(F[s].loc[dt])};z=[]
 for s,v in vals.items():
  ix=R[s].index;p=ix.get_loc(dt)
  if p+1<len(ix):z.append((v,R[s].iloc[p+1]))
 if len(z)>=8:
  q=pd.Series([a for a,b in z]).corr(pd.Series([b for a,b in z]));ics.append(q);obs.append(len(z));reg.setdefault(dt.year,[]).append(q)
  rr=pd.Series(vals).rank(pct=True)
  if prev is not None:turn.append(np.mean([abs(rr[s]-prev[s]) for s in set(rr.index)&set(prev.index)]))
  prev=rr
print('dates',len(ics),'avg_names',np.mean(obs),'assets',len(D),'total_dates',len(all_dates),'coverage',len(ics)/len(all_dates),'turnover',np.mean(turn));print('daily',stat(ics));print('regimes',{y:stat(a) for y,a in reg.items()})
