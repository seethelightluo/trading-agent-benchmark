import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date') for a in A}
# Overnight gap reversal: fade today's open relative to prior close, signal known after today's close for next-day return.
f={a:-(D[a].open/D[a].close.shift(1)-1) for a in A}
rows=[];sig=[]
for dt in sorted(set().union(*[set(D[a].index) for a in A])):
 vals={a:f[a].get(dt,np.nan) for a in A}; good=[v for v in vals.values() if np.isfinite(v)]
 if len(good)<8:continue
 for a in A:sig.append((dt,a,vals[a]))
 for a in A:
  if not np.isfinite(vals[a]) or dt not in D[a].index:continue
  ix=D[a].index.get_loc(dt)
  if ix+1<len(D[a]): pass
 z=[];y=[]
 for a in A:
  if np.isfinite(vals[a]) and dt in D[a].index:
   ix=D[a].index.get_loc(dt)
   if ix+1<len(D[a]):z.append(vals[a]);y.append(D[a].close.iloc[ix+1]/D[a].close.iloc[ix]-1)
 if len(z)>=8:rows.append((dt,spearmanr(z,y).statistic,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']);q=r.set_index('date').ic
print('dates',len(q),'avg_n',r.n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(),'hit',(q>0).mean())
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
 s=q.loc[lo:hi];print(lo,len(s),s.mean(),s.mean()/s.std())
out=pd.DataFrame(sig,columns=['date','asset','signal']);out.to_csv('../persistent/factor_signals_miner_1_20270225_overnight_fade.csv',index=False)
print('coverage',len(out)/(len(set(out.date))*15),'turn',out.pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
