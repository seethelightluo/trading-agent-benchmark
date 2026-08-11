import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]
 D[s]=x[x.close.notna()]
# Momentum acceleration: recent 20-session return relative to prior 40-session return,
# scaled by recent volatility; isolates strengthening trends while controlling risk.
for h in [5,10,20]:
 rows=[]
 for s,x in D.items():
  ix=x.index; r=x.close.pct_change()
  for j in range(60,len(ix)-h):
   recent=x.close.iloc[j]/x.close.iloc[j-20]-1
   prior=x.close.iloc[j-20]/x.close.iloc[j-60]-1
   vol=r.iloc[j-19:j+1].std(ddof=1)
   y=x.close.iloc[j+h]/x.close.iloc[j]-1
   f=(recent-prior/2)/(vol+0.02)
   if np.isfinite(f) and np.isfinite(y): rows.append((ix[j],s,f,y))
 a=pd.DataFrame(rows,columns=['date','s','f','y']); q=[]; ds=[]; ns=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:
   q.append(spearmanr(g.f,g.y).statistic); ds.append(dt); ns.append(len(g))
 z=np.array(q); print('horizon',h,'dates',len(z),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round(np.mean(z>0),4))
 if len(z): print('annual',{y:round(z[np.array([d.year for d in ds])==y].mean(),6) for y in sorted(set(d.year for d in ds))})
# rank turnover on daily valid dates
prev=None; ts=[]; valid=0
for dt in sorted(set().union(*[set(x.index) for x in D.values()])):
 vals={}
 for s,x in D.items():
  if dt not in x.index: continue
  j=x.index.get_loc(dt)
  if j<60: continue
  r=x.close.pct_change(); vol=r.iloc[j-19:j+1].std(ddof=1)
  f=((x.close.iloc[j]/x.close.iloc[j-20]-1)-(x.close.iloc[j-20]/x.close.iloc[j-60]-1)/2)/(vol+0.02)
  if np.isfinite(f): vals[s]=f
 z=pd.Series(vals)
 if len(z)>=8:
  valid+=1; rk=z.rank(pct=True)
  if prev is not None:
   ix=rk.index.intersection(prev.index); ts.append(np.abs(rk[ix]-prev[ix]).mean())
  prev=rk
print('turnover',round(np.mean(ts),6),'turnover_dates',valid)
