import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date')['close'] for a in A}
r={a:p[a].pct_change() for a in A}
# Cross-asset lead-lag: median recent 3-day return of all OTHER tradable assets.
raw={}
for a in A:
 others=pd.concat([r[b] for b in A if b!=a],axis=1).sort_index()
 daily=others.median(axis=1)
 raw[a]=daily.rolling(3,min_periods=3).mean()
idx=sorted(set().union(*[set(p[a].index) for a in A])); rows=[]; signals=[]
for d in idx:
 vals={a:raw[a].get(d,np.nan) for a in A}; good=[v for v in vals.values() if np.isfinite(v)]
 if len(good)<8: continue
 med=np.nanmedian(good)
 for a in A: signals.append((d,a,vals[a]-med if np.isfinite(vals[a]) else np.nan))
 for h in [1,5,10]:
  f=[]; y=[]
  for a in A:
   if d not in p[a].index: continue
   i=p[a].index.get_loc(d); z=vals[a]-med
   if np.isfinite(z) and i+h<len(p[a]): f.append(z); y.append(p[a].iloc[i+h]/p[a].iloc[i]-1)
  if len(f)>=8: rows.append((d,h,spearmanr(f,y).statistic,len(f)))
df=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 x=df[df.h==h]; print('H',h,'dates',len(x),'avg_n',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,4),'IC',round(x.ic.mean(),6),'ICIR',round(x.ic.mean()/x.ic.std(),6),'hit',round((x.ic>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  z=x.set_index('date').loc[lo:hi].ic; print(lo,len(z),round(z.mean(),6),round(z.mean()/z.std(),6) if len(z)>1 else np.nan)
w=pd.DataFrame(signals,columns=['date','asset','signal']); w.to_csv('../persistent/factor_signals_miner_2_20270225_crosslead.csv',index=False)
piv=w.pivot(index='date',columns='asset',values='signal'); print('turnover',round(piv.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6),'signal_rows',len(w))
