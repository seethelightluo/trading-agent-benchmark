import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}; R={a:P[a].pct_change() for a in A}
# Volatility-scaled cross-sectional mean-reversion: recent 3d reversal divided by 20d risk.
F={a:-(P[a].pct_change(3)/(R[a].rolling(20,min_periods=15).std()*np.sqrt(3)+1e-8)) for a in A}
rows=[];sig=[]
for d in sorted(set().union(*[set(x.index) for x in P.values()])):
 v={a:F[a].get(d,np.nan) for a in A}; g=[x for x in v.values() if np.isfinite(x)]; med=np.nanmedian(g) if len(g)>=8 else np.nan
 for a in A: sig.append((d,a,v[a]-med if np.isfinite(v[a]) and np.isfinite(med) else np.nan))
 for h in [1,5,10]:
  x=[];y=[]
  for a in A:
   z=v[a]-med if np.isfinite(v[a]) and np.isfinite(med) else np.nan
   if d in P[a].index and np.isfinite(z):
    i=P[a].index.get_loc(d)
    if i+h<len(P[a]): x.append(z);y.append(P[a].iloc[i+h]/P[a].iloc[i]-1)
  if len(x)>=8: rows.append((d,h,spearmanr(x,y).statistic,len(x)))
df=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 z=df[df.h==h];print('H',h,'dates',len(z),'avg_n',round(z.n.mean(),2),'coverage',round(z.n.mean()/15,4),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(),6),'hit',round((z.ic>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  q=z.set_index('date').loc[lo:hi].ic;print(lo,len(q),round(q.mean(),6),round(q.mean()/q.std(),6) if len(q)>1 else None)
out=pd.DataFrame(sig,columns=['date','asset','signal']);out.to_csv('../persistent/factor_signals_miner_1_20270225_volscaled_rev3.csv',index=False);w=out.pivot(index='date',columns='asset',values='signal');print('turnover',round(w.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
