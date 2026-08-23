import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=4000) for s in U}; D={s:x.sort_values('date').drop_duplicates('date').set_index('date') for s,x in D.items() if x is not None and len(x)>80}
def sig(x,i):
 c=x.close.astype(float)
 if i<25:return np.nan
 r=c.iloc[i]/c.iloc[i-40]-1
 q=c.pct_change().iloc[i-39:i+1].dropna(); down=q[q<0]
 ds=np.sqrt((down**2).mean()) if len(down) else q.std()
 return r/(ds*np.sqrt(20)+.01)
dates=sorted(set().union(*[set(x.index) for x in D.values()])); rows=[]
for dt in dates:
 z=[]; y=[]
 for s,x in D.items():
  p=x.index.searchsorted(dt)
  if p>=len(x) or x.index[p]!=dt or p+10>=len(x):continue
  a=sig(x,p)
  if np.isfinite(a):z.append(a);y.append(x.close.iloc[p+10]/x.close.iloc[p]-1)
 if len(z)>=8: rows.append((dt,pd.Series(z).corr(pd.Series(y),method='spearman'),len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).dropna()
print('candidate=downside_adjusted_momentum_40d dates=%d avg_n=%.2f coverage=%.4f assets=%d'%(len(r),r.n.mean(),len(r)/len(dates),len(D)))
print('IC=%.8f ICIR=%.8f hit=%.6f'%(r.ic.mean(),r.ic.mean()/r.ic.std(ddof=1),(r.ic>0).mean()))
for nm,a,b in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-12-31'),('2027-28','2027-01-01','2028-12-31'),('2029YTD','2029-01-01','2029-07-11')]:
 q=r[(r.date>=a)&(r.date<=b)]; print(nm,len(q),'ic=%.8f icir=%.5f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)) if len(q)>2 else 'NA')
for h in [1,5,10,20]:
 a=[]
 for dt in dates:
  z=[];y=[]
  for s,x in D.items():
   p=x.index.searchsorted(dt)
   if p>=len(x) or x.index[p]!=dt or p+h>=len(x):continue
   v=sig(x,p)
   if np.isfinite(v):z.append(v);y.append(x.close.iloc[p+h]/x.close.iloc[p]-1)
  if len(z)>=8:a.append(pd.Series(z).corr(pd.Series(y),method='spearman'))
 a=pd.Series(a).dropna();print('decay_%dd ic=%.8f n=%d'%(h,a.mean(),len(a)))
