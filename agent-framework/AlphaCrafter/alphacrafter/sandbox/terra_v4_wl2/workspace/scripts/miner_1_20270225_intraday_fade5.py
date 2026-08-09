import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date') for a in A}
# Factor: fade persistent intraday close-open pressure over 5 sessions; all inputs lagged at decision close.
x={a:(D[a].close/D[a].open-1).rolling(5,min_periods=5).mean() for a in A}
f={a:-x[a] for a in A}
rows=[]; sig=[]
for dt in sorted(set().union(*[set(D[a].index) for a in A])):
 vals={a:(f[a].get(dt,np.nan)) for a in A}; good=[v for v in vals.values() if np.isfinite(v)]
 if len(good)<8: continue
 for a in A: sig.append((dt,a,vals[a]))
 for h in [1,3,5]:
  z=[]; y=[]
  for a in A:
   if not np.isfinite(vals[a]) or dt not in D[a].index: continue
   ix=D[a].index.get_loc(dt)
   if ix+h>=len(D[a]): continue
   z.append(vals[a]); y.append(D[a].close.iloc[ix+h]/D[a].close.iloc[ix]-1)
  if len(z)>=8: rows.append((dt,h,spearmanr(z,y).statistic,len(z)))
r=pd.DataFrame(rows,columns=['date','h','ic','n'])
print('factor=5d negative intraday return (close/open), dates/instruments by horizon')
for h in [1,3,5]:
 q=r[r.h==h].set_index('date').ic
 print(h,len(q),round(r[r.h==h].n.mean(),2),round(q.mean(),6),round(q.mean()/q.std(),6),round((q>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  s=q.loc[lo:hi]; print(' ',lo,len(s),round(s.mean(),6) if len(s) else None,round(s.mean()/s.std(),4) if len(s)>1 else None)
out=pd.DataFrame(sig,columns=['date','asset','signal']);out.to_csv('../persistent/factor_signals_miner_1_20270225_intraday_fade5.csv',index=False)
rank=out.pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True)
print('coverage',len(out)/ (len(set(out.date))*15),'turnover',rank.diff().abs().mean(axis=1).mean())
