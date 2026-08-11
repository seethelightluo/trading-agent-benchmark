import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
# Per-asset positional windows preserve each instrument's own trading calendar.
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut] for s in U}
for s in U: D[s]=D[s][D[s].close.notna()]
F={h:[] for h in [5,10,20]}; dates=sorted(set().union(*[set(x.index) for x in D.values()]))
for d in dates:
 for h in F:
  vals=[]; ys=[]
  for s,x in D.items():
   if d not in x.index: continue
   j=x.index.get_loc(d)
   if j<20 or j+h>=len(x): continue
   r=x.close.pct_change().iloc[j-19:j+1]
   path=r.abs().sum(); net=x.close.iloc[j]/x.close.iloc[j-20]-1
   y=x.close.iloc[j+h]/x.close.iloc[j]-1
   if np.isfinite(net) and np.isfinite(path) and np.isfinite(y): vals.append(net/(path+1e-8));ys.append(y)
  if len(vals)>=8 and np.std(vals)>0 and np.std(ys)>0: F[h].append((d,len(vals),spearmanr(vals,ys).statistic))
for h,a0 in F.items():
 a=np.array([z[2] for z in a0]); n=np.array([z[1] for z in a0]); print('horizon',h,'dates',len(a),'avgN',round(n.mean(),2),'coverage',round(n.mean()/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 if h==10: print('annual',{y:round(a[[d.year==y for d,_,_ in a0]].mean(),6) for y in sorted(set(d.year for d,_,_ in a0))})
# rank turnover on common dates
prev=None; ts=[]
for d in dates:
 vals={}
 for s,x in D.items():
  if d not in x.index: continue
  j=x.index.get_loc(d)
  if j>=20:
   rr=x.close.pct_change().iloc[j-19:j+1]; vals[s]=(x.close.iloc[j]/x.close.iloc[j-20]-1)/(rr.abs().sum()+1e-8)
 z=pd.Series(vals).dropna()
 if len(z)>=8:
  rk=z.rank(pct=True)
  if prev is not None:
   ix=rk.index.intersection(prev.index);ts.append(np.abs(rk[ix]-prev[ix]).mean())
  prev=rk
print('turnover',round(np.mean(ts),6),'valid_dates',len(ts),'coverage_dates',round(len(ts)/len(dates),4))
