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
 c=d.close.astype(float); r=c.pct_change()
 # trend-consistency momentum: signed 20d return multiplied by fraction of positive sessions,
 # volatility normalized, lagged to avoid look-ahead.
 trend=c.pct_change(20); consistency=(r>0).rolling(20,min_periods=15).mean()
 vol=r.rolling(20,min_periods=15).std()
 F[s]=(trend*consistency/vol).shift(1);R[s]=r
all_dates=sorted(set().union(*[set(x.index) for x in F.values()]));ics=[];obs=[];turn=[];prev=None;dec={1:[],5:[],10:[]};reg={}
for dt in all_dates:
 vals={s:F[s].loc[dt] for s in F if dt in F[s].index and np.isfinite(F[s].loc[dt])}; ys={1:{},5:{},10:{}}
 for s in vals:
  ix=R[s].index;p=ix.get_loc(dt)
  for h in ys:
   if p+h<len(ix):ys[h][s]=R[s].iloc[p+1:p+h+1].sum()
 z=[(v,ys[1][s]) for s,v in vals.items() if s in ys[1]]
 if len(z)>=8:
  q=pd.Series([a for a,b in z]).corr(pd.Series([b for a,b in z]));ics.append(q);obs.append(len(z));reg.setdefault(dt.year,[]).append(q)
  for h in dec:
   zz=[(v,ys[h][s]) for s,v in vals.items() if s in ys[h]];dec[h].append(pd.Series([a for a,b in zz]).corr(pd.Series([b for a,b in zz])) if len(zz)>=8 else np.nan)
  rr=pd.Series(vals).rank(pct=True)
  if prev is not None: turn.append(np.mean([abs(rr[s]-prev[s]) for s in set(rr.index)&set(prev.index)]))
  prev=rr
def st(a):
 a=np.asarray(a,float);a=a[np.isfinite(a)];return len(a),float(np.mean(a)),float(np.mean(a)/np.std(a,ddof=1)),float(np.mean(a>0))
print('dates',len(ics),'avg_names',np.mean(obs),'assets',len(D),'total_dates',len(all_dates),'coverage',len(ics)/len(all_dates),'turnover',np.mean(turn));print('daily',st(ics))
for h,a in dec.items():print(str(h)+'d',st(a))
print('regimes',{y:st(a) for y,a in reg.items()})
