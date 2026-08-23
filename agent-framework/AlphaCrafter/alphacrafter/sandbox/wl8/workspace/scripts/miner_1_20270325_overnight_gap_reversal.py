import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; cutoff=pd.Timestamp('2027-03-24')
def calc(h=1):
 xs=[]
 for s in U:
  d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).sort_values('date'); d=d[d.date<=cutoff].copy()
  d['signal']=-(d.open/d.close.shift(1)-1); d['fwd']=d.close.shift(-h)/d.close-1
  xs.append(d[['date','signal','fwd']].assign(symbol=s))
 z=pd.concat(xs).dropna(); ics=[]; rows=[]
 for dt,g in z.groupby('date'):
  if len(g)>=8 and g.signal.nunique()>1 and g.fwd.nunique()>1:
   ic=spearmanr(g.signal,g.fwd).statistic
   if np.isfinite(ic): ics.append(ic); rows.append((dt,ic,len(g)))
 return np.array(ics),rows,len(z),len(set(z.date))
a,rows,n,nd=calc(); print('cutoff',cutoff.date(),'dates',len(a),'rows',n,'avg_names',n/nd,'coverage',n/(len(U)*nd))
print('IC %.8f ICIR %.8f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)))
for label,lo,hi in [('2020-2024','2020-01-01','2025-01-01'),('2025-2026','2025-01-01','2027-01-01'),('2027','2027-01-01','2027-03-25')]:
 q=np.array([ic for dt,ic,nm in rows if pd.Timestamp(lo)<=dt<pd.Timestamp(hi)])
 print(label,len(q), 'IC %.6f ICIR %.6f hit %.3f'%(q.mean(),q.mean()/q.std(ddof=1),np.mean(q>0)) if len(q)>1 else '')
for h in [2,5,10]:
 ii,_,_,_=calc(h); print('horizon',h,'IC %.6f ICIR %.6f'%(ii.mean(),ii.mean()/ii.std(ddof=1)))
