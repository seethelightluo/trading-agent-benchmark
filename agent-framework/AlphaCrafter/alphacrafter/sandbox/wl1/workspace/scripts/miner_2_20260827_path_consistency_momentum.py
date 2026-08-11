import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut] for s in U}
# Path-consistency momentum: cumulative return scaled by fraction of positive daily returns,
# with a mild penalty for sign inconsistency; all inputs lagged at decision date.
for w in [10,20,40]:
 for h in [5,10,20]:
  rows=[]
  dates=sorted(set().union(*[set(x.index) for x in D.values()]))
  for d in dates:
   vals=[]; ys=[]
   for s,x in D.items():
    if d not in x.index: continue
    j=x.index.get_loc(d)
    if j<w or j+h>=len(x): continue
    r=x.close.pct_change().iloc[j-w+1:j+1].dropna()
    if len(r)<w-1: continue
    mom=x.close.iloc[j]/x.close.iloc[j-w]-1
    consistency=(r>0).mean()-(r<0).mean()
    f=mom*consistency
    y=x.close.iloc[j+h]/x.close.iloc[j]-1
    if np.isfinite(f) and np.isfinite(y): vals.append(f);ys.append(y)
   if len(vals)>=8 and np.std(vals)>0 and np.std(ys)>0: rows.append((d,len(vals),spearmanr(vals,ys).statistic))
  a=np.array([q[2] for q in rows]); n=np.array([q[1] for q in rows])
  years=sorted(set(q[0].year for q in rows))
  print('w',w,'h',h,'dates',len(a),'avgN',round(n.mean(),2),'coverage',round(n.mean()/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4),'annual',{y:round(a[[q[0].year==y for q in rows]].mean(),5) for y in years})
# rank turnover for w20
prev=None; ts=[]; count=0
for d in sorted(set().union(*[set(x.index) for x in D.values()])):
 vals={}
 for s,x in D.items():
  if d not in x.index: continue
  j=x.index.get_loc(d)
  if j<20: continue
  r=x.close.pct_change().iloc[j-19:j+1].dropna(); mom=x.close.iloc[j]/x.close.iloc[j-20]-1
  vals[s]=mom*((r>0).mean()-(r<0).mean())
 z=pd.Series(vals).dropna()
 if len(z)>=8:
  rk=z.rank(pct=True)
  if prev is not None:
   ix=rk.index.intersection(prev.index); ts.append(np.abs(rk[ix]-prev[ix]).mean())
  prev=rk; count+=1
print('turnover',round(float(np.mean(ts)),6),'turnover_dates',len(ts),'factor_dates',count)
