import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
xs=[]
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).sort_values('date'); d['r']=d.close.pct_change(10); xs.append(d[['date','r']].assign(symbol=s))
z=pd.concat(xs).dropna(); vals=[]; rows=[]
for dt,g in z.groupby('date'):
 if len(g)>=8:
  g=g.copy(); g['sig']=g.r-g.r.median(); # relative 10d momentum
  # forward 1d from current close
  f=[]
  for s in g.symbol:
   d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).sort_values('date'); x=d.index[d.date==dt]
   if len(x) and x[0]+1<len(d): f.append((s,d.close.iloc[x[0]+1]/d.close.iloc[x[0]]-1))
  ff=pd.DataFrame(f,columns=['symbol','f']); q=g.merge(ff,on='symbol').dropna()
  if len(q)>=8 and q.sig.nunique()>1 and q.f.nunique()>1:
   ic=spearmanr(q.sig,q.f).statistic
   if np.isfinite(ic): vals.append(ic); rows.append((dt,ic,len(q)))
a=np.array(vals); print('dates',len(a),'rows',len(z),'avg_names',np.mean([x[2] for x in rows]),'coverage',len(z)/(len(U)*len(set(z.date))))
print('IC %.8f ICIR %.8f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)))
for label,lo,hi in [('2020-24','2020-01-01','2025-01-01'),('2025-26','2025-01-01','2027-01-01')]:
 q=np.array([ic for dt,ic,n in rows if pd.Timestamp(lo)<=dt<pd.Timestamp(hi)]); print(label,len(q),'IC %.6f ICIR %.6f hit %.3f'%(q.mean(),q.mean()/q.std(ddof=1),np.mean(q>0)))
