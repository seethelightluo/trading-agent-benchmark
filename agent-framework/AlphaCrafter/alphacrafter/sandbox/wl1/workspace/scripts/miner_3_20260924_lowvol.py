import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15'); D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]; D[s]=x[x.close.notna()]
# Candidate: low-volatility quality, inverse of 20-session realized volatility, cross-sectionally rankable.
for h in [5,10,20,30]:
 rows=[]
 for s,x in D.items():
  r=x.close.pct_change()
  for j in range(30,len(x)-h):
   v=r.iloc[j-19:j+1].std()
   f=-v; y=x.close.iloc[j+h]/x.close.iloc[j]-1
   if np.isfinite(f) and np.isfinite(y): rows.append((x.index[j],s,f,y))
 a=pd.DataFrame(rows,columns=['date','s','f','y']); q=[]; ds=[]; ns=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:
   q.append(spearmanr(g.f,g.y).statistic); ds.append(dt); ns.append(len(g))
 z=np.asarray(q); print('horizon',h,'dates',len(z),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round(np.mean(z>0),4))
 print('annual',{y:round(z[np.array([d.year for d in ds])==y].mean(),6) for y in sorted(set(d.year for d in ds))})
