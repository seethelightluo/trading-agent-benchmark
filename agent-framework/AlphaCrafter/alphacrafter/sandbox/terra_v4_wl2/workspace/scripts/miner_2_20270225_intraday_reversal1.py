import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date') for a in A}
# Intraday reversal: fade each asset's completed session return, cross-sectionally demeaned.
raw={a:-(p[a].close/p[a].open-1) for a in A}; idx=sorted(set().union(*[set(x.index) for x in p.values()])); rows=[]; sig=[]
for d in idx:
 vals={a:raw[a].get(d,np.nan) for a in A}; good=[v for v in vals.values() if np.isfinite(v)]
 if len(good)<8: continue
 med=np.median(good)
 for a in A:
  z=vals[a]-med if np.isfinite(vals[a]) else np.nan; sig.append((d,a,z))
 for h in (1,5,10):
  f=[];y=[]
  for a in A:
   if d not in p[a].index or not np.isfinite(vals[a]): continue
   i=p[a].index.get_loc(d)
   if i+h<len(p[a]): f.append(vals[a]-med); y.append(p[a].close.iloc[i+h]/p[a].close.iloc[i]-1)
  if len(f)>=8: rows.append((d,h,spearmanr(f,y).statistic,len(f)))
df=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in (1,5,10):
 x=df[df.h==h]; print('H',h,'dates',len(x),'avg_n',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,4),'IC',round(x.ic.mean(),6),'ICIR',round(x.ic.mean()/x.ic.std(),6),'hit',round((x.ic>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  z=x.set_index('date').loc[lo:hi].ic; print(lo,len(z),round(z.mean(),6),round(z.mean()/z.std(),6) if len(z)>1 else np.nan)
w=pd.DataFrame(sig,columns=['date','asset','signal']); w.to_csv('../persistent/factor_signals_miner_2_20270225_intraday_reversal1.csv',index=False)
print('turnover',round(w.pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
