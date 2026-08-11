import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut] for s in U}
# Momentum breadth: reward aligned 5d and 20d trends, penalize disagreement, scaled by recent volatility.
def factor(x,j):
 c=x.close
 r5=c.iloc[j]/c.iloc[j-5]-1; r20=c.iloc[j]/c.iloc[j-20]-1
 vol=c.pct_change().iloc[j-19:j+1].std()*np.sqrt(20)
 align=np.sign(r5*r20)
 return (0.4*r5+0.6*r20)*align/(vol+1e-8)
rows=[]
for d in sorted(set().union(*[set(x.index) for x in D.values()])):
 vals=[]; ys=[]
 for s,x in D.items():
  if d not in x.index: continue
  j=x.index.get_loc(d)
  if j<20 or j+10>=len(x): continue
  v=factor(x,j); y=x.close.iloc[j+10]/x.close.iloc[j]-1
  if np.isfinite(v) and np.isfinite(y): vals.append(v);ys.append(y)
 if len(vals)>=8 and np.std(vals)>0 and np.std(ys)>0: rows.append((d,len(vals),spearmanr(vals,ys).statistic))
a=np.array([z[2] for z in rows]); n=np.array([z[1] for z in rows])
print('dates',len(a),'avgN',round(n.mean(),3),'coverage',round(n.mean()/15,5),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),5))
print('annual',{y:round(a[[d.year==y for d,_,_ in rows]].mean(),6) for y in sorted(set(d.year for d,_,_ in rows))})
# decay on same signal
for h in [1,5,10,20]:
 z=[]
 for d in sorted(set().union(*[set(x.index) for x in D.values()])):
  vs=[]; ys=[]
  for s,x in D.items():
   if d not in x.index: continue
   j=x.index.get_loc(d)
   if j<20 or j+h>=len(x): continue
   v=factor(x,j); y=x.close.iloc[j+h]/x.close.iloc[j]-1
   if np.isfinite(v) and np.isfinite(y): vs.append(v);ys.append(y)
  if len(vs)>=8 and np.std(vs)>0 and np.std(ys)>0:z.append(spearmanr(vs,ys).statistic)
 print('decay',h,round(np.mean(z),6),len(z))
# rank turnover
prev=None; ts=[]
for d in sorted(set().union(*[set(x.index) for x in D.values()])):
 vals={}
 for s,x in D.items():
  if d in x.index and x.index.get_loc(d)>=20: vals[s]=factor(x,x.index.get_loc(d))
 z=pd.Series(vals).dropna()
 if len(z)>=8:
  rk=z.rank(pct=True)
  if prev is not None:
   ix=rk.index.intersection(prev.index);ts.append(np.abs(rk[ix]-prev[ix]).mean())
  prev=rk
print('turnover',round(np.mean(ts),6),'valid',len(ts))
