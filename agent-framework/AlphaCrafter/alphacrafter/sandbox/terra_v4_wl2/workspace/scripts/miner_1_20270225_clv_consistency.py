import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date') for a in A}
# CLV consistency: mean close-location value over the prior 10 completed sessions,
# rewarding persistent closes near the high (and penalizing closes near the low).
F={}; R={}
for a,d in P.items():
 rng=(d.high-d.low).replace(0,np.nan)
 clv=(2*d.close-d.high-d.low)/rng
 F[a]=clv.rolling(10,min_periods=8).mean().shift(1)
 R[a]=d.close.pct_change()
rows=[]; sig=[]
for date in sorted(set().union(*[set(x.index) for x in P.values()])):
 vals={a:F[a].get(date,np.nan) for a in A}; good=[x for x in vals.values() if np.isfinite(x)]
 if len(good)<8: continue
 med=np.nanmedian(good)
 for a in A: sig.append((date,a,vals[a]-med if np.isfinite(vals[a]) else np.nan))
 for h in (1,5,10):
  f=[]; y=[]
  for a in A:
   if date not in P[a].index: continue
   z=vals[a]-med; i=P[a].index.get_loc(date)
   if np.isfinite(z) and i+h<len(P[a]):
    f.append(z); y.append(P[a].close.iloc[i+h]/P[a].close.iloc[i]-1)
  if len(f)>=8: rows.append((date,h,spearmanr(f,y).statistic,len(f)))
df=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in (1,5,10):
 x=df[df.h==h]; print('H',h,'dates',len(x),'avg_n',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,4),'IC',round(x.ic.mean(),6),'ICIR',round(x.ic.mean()/x.ic.std(),6),'hit',round((x.ic>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  z=x.set_index('date').loc[lo:hi].ic; print('REG',lo,len(z),round(z.mean(),6),round(z.mean()/z.std(),6) if len(z)>1 else np.nan)
w=pd.DataFrame(sig,columns=['date','asset','signal']).pivot(index='date',columns='asset',values='signal'); print('turnover',round(w.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
pd.DataFrame(sig,columns=['date','asset','signal']).to_csv('../persistent/factor_signals_miner_1_20270225_clv_consistency.csv',index=False)
