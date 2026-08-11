import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]
 D[s]=x[x.close.notna()]
# Candidate: range-adjusted directional pressure. Average close-location value over 20d
# is weighted by upside participation and inversely by realized volatility.
for w in [10,20,40]:
 for h in [5,10,20]:
  rows=[]
  for s,x in D.items():
   ix=x.index; r=x.close.pct_change()
   clv=((2*x.close-x.high-x.low)/(x.high-x.low).replace(0,np.nan)).clip(-1,1)
   for j in range(w,len(ix)-h):
    rr=r.iloc[j-w+1:j+1].to_numpy(); c=clv.iloc[j-w+1:j+1].to_numpy()
    y=x.close.iloc[j+h]/x.close.iloc[j]-1
    if np.isfinite(rr).all() and np.isfinite(c).all() and np.isfinite(y):
     up=np.mean(rr>0); vol=np.std(rr,ddof=1)
     f=np.mean(c)*up/(vol+1e-6)
     rows.append((ix[j],s,f,y))
  a=pd.DataFrame(rows,columns=['date','s','f','y']); q=[]; ds=[]; ns=[]
  for dt,g in a.groupby('date'):
   if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:
    q.append(spearmanr(g.f,g.y).statistic); ds.append(dt); ns.append(len(g))
  z=np.array(q); print('window',w,'horizon',h,'dates',len(z),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round(np.mean(z>0),4))
  if h==10 and len(z): print('annual',{y:round(z[np.array([d.year for d in ds])==y].mean(),6) for y in sorted(set(d.year for d in ds))})
# turnover selected 20d
prev=None; ts=[]; valid=0
for dt in sorted(set().union(*[set(x.index) for x in D.values()])):
 vals={}
 for s,x in D.items():
  if dt not in x.index: continue
  j=x.index.get_loc(dt)
  if j<20: continue
  r=x.close.pct_change().iloc[j-19:j+1].to_numpy(); c=((2*x.close-x.high-x.low)/(x.high-x.low).replace(0,np.nan)).iloc[j-19:j+1].to_numpy()
  if np.isfinite(r).all() and np.isfinite(c).all(): vals[s]=np.mean(c)*np.mean(r>0)/(np.std(r,ddof=1)+1e-6)
 z=pd.Series(vals)
 if len(z)>=8:
  valid+=1; rk=z.rank(pct=True)
  if prev is not None:
   ix=rk.index.intersection(prev.index); ts.append(np.abs(rk[ix]-prev[ix]).mean())
  prev=rk
print('turnover',round(np.mean(ts),6),'turnover_dates',valid)
