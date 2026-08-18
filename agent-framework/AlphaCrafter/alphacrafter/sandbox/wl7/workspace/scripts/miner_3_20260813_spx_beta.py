import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   x=fn(s,2400)
   if x is not None and len(x)>100:return x
  except:pass
D={s:g(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
R={}
for s,d in D.items():
 d=d.copy();d.date=pd.to_datetime(d.date).dt.normalize(); d=d.drop_duplicates('date').set_index('date').sort_index();R[s]=d.close.pct_change()
bench=R['SPX']; F={}
for s,r in R.items():
 # negative rolling beta to SPX, defensive assets receive high scores
 cov=r.rolling(60,min_periods=45).cov(bench); var=bench.rolling(60,min_periods=45).var();F[s]=(-cov/var)
Ds=sorted(set().union(*[set(x.index) for x in F.values()])); out={k:[] for k in [1,5,10]}; ns=[]
for dt in Ds:
 v={s:F[s].loc[dt] for s in F if dt in F[s].index and pd.notna(F[s].loc[dt])}; ys={k:{} for k in out}
 for s in v:
  ix=R[s].index
  if dt not in ix: continue
  p=ix.get_loc(dt)
  for k in out:
   if p+k<len(ix):ys[k][s]=R[s].iloc[p+1:p+k+1].sum()
 for k in out:
  z=[(v[s],ys[k][s]) for s in v if s in ys[k]]
  if len(z)>=8:out[k].append(pd.Series([a for a,b in z]).corr(pd.Series([b for a,b in z])))
 ns.append(len(v))
def st(x):
 x=np.array(x);return len(x),np.nanmean(x),np.nanmean(x)/np.nanstd(x,ddof=1)
print('dates',len(out[1]),'avg names',np.mean(ns),'daily',st(out[1]),'hit',np.mean(np.array(out[1])>0))
for k in out:print(k,st(out[k]))
for a,b in [(2020,2022),(2023,2024),(2025,2026)]:
 # approximate date filtered correctly
 z=[]
 for dt in Ds:
  if a<=dt.year<=b:
   v={s:F[s].loc[dt] for s in F if dt in F[s].index and pd.notna(F[s].loc[dt])};ixs=[]
   for s in v:
    ix=R[s].index
  if dt not in ix: continue
  p=ix.get_loc(dt)
    if p+1<len(ix):ixs.append((v[s],R[s].iloc[p+1]))
   if len(ixs)>=8:z.append(pd.Series([x[0] for x in ixs]).corr(pd.Series([x[1] for x in ixs])))
 print(a,b,st(z))
