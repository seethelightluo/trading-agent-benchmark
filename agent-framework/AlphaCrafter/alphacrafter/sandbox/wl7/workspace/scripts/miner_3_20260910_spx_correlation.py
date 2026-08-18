import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   x=fn(s,2400)
   if x is not None and len(x)>150:return x
  except:pass
D={}
for s in U:
 x=fetch(s)
 if x is not None:
  x=x.copy();x.date=pd.to_datetime(x.date).dt.normalize();D[s]=x.drop_duplicates('date').set_index('date').sort_index()
R={s:d.close.astype(float).pct_change() for s,d in D.items()}; benchmark=R['SPX']
F={}
for s,r in R.items():
 z=pd.concat([r,benchmark],axis=1,keys=['a','b']).dropna()
 cov=z.a.rolling(60,min_periods=45).cov(z.b); var=z.b.rolling(60,min_periods=45).var()
 # low correlation, shifted to prevent same-day use
 F[s]=(-cov/var).shift(1)
all_dates=sorted(set().union(*[set(v.index) for v in F.values()]));ics=[];obs=[];turn=[];prev=None;reg={};hs={5:[],10:[]}
for dt in all_dates:
 vals={s:F[s].loc[dt] for s in F if dt in F[s].index and np.isfinite(F[s].loc[dt])}; ys={1:{},5:{},10:{}}
 for s in vals:
  ix=R[s].index;p=ix.get_loc(dt)
  for k in ys:
   if p+k<len(ix):ys[k][s]=R[s].iloc[p+1:p+k+1].sum()
 z=[(vals[s],ys[1][s]) for s in vals if s in ys[1]]
 if len(z)>=8:
  q=pd.Series([a for a,b in z]).corr(pd.Series([b for a,b in z]));ics.append(q);obs.append(len(z))
  for k in hs:
   zz=[(vals[s],ys[k][s]) for s in vals if s in ys[k]];hs[k].append(pd.Series([a for a,b in zz]).corr(pd.Series([b for a,b in zz])) if len(zz)>=8 else np.nan)
  rr=pd.Series(vals).rank(pct=True); rr.index=rr.index.astype(str)
  if prev is not None: turn.append(np.mean([abs(rr.loc[s]-prev.loc[s]) for s in set(rr.index)&set(prev.index)]))
  prev=rr;reg.setdefault(dt.year,[]).append(q)
def st(x):
 x=np.array(x);x=x[np.isfinite(x)];return len(x),float(np.mean(x)),float(np.mean(x)/np.std(x,ddof=1)),float(np.mean(x>0))
print('dates',len(ics),'avg_names',np.mean(obs),'coverage',len(ics)/len(all_dates),'turnover',np.mean(turn));print('daily',st(ics))
for k,v in hs.items():print(str(k)+'d',st(v))
print('regimes',{k:st(v) for k,v in reg.items()})
