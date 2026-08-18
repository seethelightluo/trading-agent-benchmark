import numpy as np, pandas as pd
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
  x=x.copy();x.date=pd.to_datetime(x.date).dt.normalize();x=x.drop_duplicates('date').set_index('date').sort_index();D[s]=x
# candidate: recovery-adjusted drawdown. Assets near a long high but with positive recent recovery rank higher;
# use distance from 60d high, normalized by 20d volatility, with a 5d recovery term.
F={};R={}
for s,d in D.items():
 c=d.close.astype(float); r=c.pct_change()
 dd=c/c.rolling(60,min_periods=45).max()-1
 vol=r.rolling(20,min_periods=15).std()
 rec=c/c.shift(5)-1
 F[s]=(dd/vol + 0.5*rec/vol).shift(1)
 R[s]=r
all_dates=sorted(set().union(*[set(v.index) for v in F.values()]))
ics=[]; obs=[]; turns=[]; rankprev=None; horizons={5:[],10:[]}; reg={}
for dt in all_dates:
 vals={s:F[s].loc[dt] for s in F if dt in F[s].index and np.isfinite(F[s].loc[dt])}
 ys={1:{},5:{},10:{}}
 for s in vals:
  ix=R[s].index;p=ix.get_loc(dt)
  for k in ys:
   if p+k<len(ix):ys[k][s]=R[s].iloc[p+1:p+k+1].sum()
 z=[(vals[s],ys[1][s]) for s in vals if s in ys[1] and np.isfinite(ys[1][s])]
 if len(z)>=8:
  q=pd.Series([a for a,b in z]).corr(pd.Series([b for a,b in z]));ics.append(q);obs.append(len(z))
  for k in (5,10):
   zz=[(vals[s],ys[k][s]) for s in vals if s in ys[k]]
   horizons[k].append(pd.Series([a for a,b in zz]).corr(pd.Series([b for a,b in zz])) if len(zz)>=8 else np.nan)
  rr=pd.Series(vals).rank(pct=True)
  if rankprev is not None:
   turns.append(np.mean([abs(rr.get(s,np.nan)-rankprev.get(s,np.nan)) for s in set(rr.index)&set(rankprev)]))
  rankprev=rr
  reg.setdefault(dt.year,[]).append(q)
def stat(x):
 x=np.array(x,dtype=float);x=x[np.isfinite(x)];return (len(x),np.mean(x),np.mean(x)/np.std(x,ddof=1) if len(x)>1 else np.nan,np.mean(x>0))
print('dates',len(ics),'avg_names',np.mean(obs),'coverage',len(ics)/len(all_dates),'turnover',np.mean(turns))
print('daily n IC ICIR hit',stat(ics))
for k,v in horizons.items():print(str(k)+'d n IC ICIR hit',stat(v))
print('regimes', {y:stat(v) for y,v in reg.items()})
print('valid_assets',len(D),'total_dates',len(all_dates))
