import pandas as pd,numpy as np, os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.sort_index() for a in A}
# Conditional oversold rebound: 5-day reversal, activated only when 20-day drawdown <= -5%.
sig={}
for a in A:
 s=p[a].pct_change(5); dd=p[a]/p[a].rolling(20,min_periods=20).max()-1
 sig[a]=(-s).where(dd<=-0.05)
rows=[]
for d in sorted(set().union(*[x.index for x in sig.values()])):
 z=[]
 for a in A:
  if d in sig[a].index:
   z.append((sig[a].loc[d],p[a].pct_change().shift(-1).loc[d]))
 z=pd.DataFrame(z,columns=['x','y']).dropna()
 if len(z)>=8 and z.x.nunique()>1 and z.y.nunique()>1:
  q=spearmanr(z.x,z.y).statistic
  if np.isfinite(q):rows.append((d,q,len(z)))
a=np.array([x[1] for x in rows]); print('dates',len(a),'avgN',np.mean([x[2] for x in rows]),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'coverage_dates',len(a)/len(p[A[0]]))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2027-02-24')]:
 q=np.array([x[1] for x in rows if lo<=str(x[0].date())<=hi]);print(lo,hi,len(q),q.mean() if len(q) else np.nan,q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
out=pd.DataFrame([(d,A[j],sig[A[j]].get(d, np.nan)) for d in sorted(set().union(*[x.index for x in sig.values()])) for j in range(len(A))],columns=['date','symbol','signal']);out.to_csv('../persistent/factor_signals_miner_2_20270225_conditional_rebound5.csv',index=False)
print('coverage values',out.signal.notna().mean(),'turnover',out.pivot(index='date',columns='symbol',values='signal').rank(pct=True).diff().abs().mean(axis=1).mean())
