import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut] for s in U}
for s in U: D[s]=D[s][D[s].close.notna()]
for w in [10,20,40]:
 for h in [5,10,20]:
  rows=[]
  for d in sorted(set().union(*[set(x.index) for x in D.values()])):
   vals=[];ys=[]
   for s,x in D.items():
    if d not in x.index: continue
    j=x.index.get_loc(d)
    if j<w or j+h>=len(x): continue
    ret=x.close.pct_change().iloc[j-w+1:j+1]
    vol=ret.std(ddof=1)*np.sqrt(w)
    mom=x.close.iloc[j]/x.close.iloc[j-w]-1
    y=x.close.iloc[j+h]/x.close.iloc[j]-1
    if np.isfinite(mom) and np.isfinite(vol) and np.isfinite(y) and vol>0: vals.append(mom/(vol+1e-8));ys.append(y)
   if len(vals)>=8 and np.std(vals)>0 and np.std(ys)>0: rows.append((d,len(vals),spearmanr(vals,ys).statistic))
  a=np.array([q[2] for q in rows]); n=np.array([q[1] for q in rows]); print('w',w,'h',h,'dates',len(a),'avgN',round(n.mean(),2),'coverage',round(n.mean()/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4),'annual',{y:round(a[[q[0].year==y for q in rows]].mean(),5) for y in sorted(set(q[0].year for q in rows))})
 # turnover at w=20 only
 if w==20:
  prev=None;ts=[]
  for d in sorted(set().union(*[set(x.index) for x in D.values()])):
   vals={}
   for s,x in D.items():
    if d in x.index:
     j=x.index.get_loc(d)
     if j>=w:
      rr=x.close.pct_change().iloc[j-w+1:j+1]; v=rr.std(ddof=1)*np.sqrt(w)
      if v>0: vals[s]=(x.close.iloc[j]/x.close.iloc[j-w]-1)/(v+1e-8)
   z=pd.Series(vals).dropna()
   if len(z)>=8:
    rk=z.rank(pct=True)
    if prev is not None:
     ix=rk.index.intersection(prev.index);ts.append(np.abs(rk[ix]-prev[ix]).mean())
    prev=rk
  print('turnover',round(np.mean(ts),6),'turnover_dates',len(ts))
