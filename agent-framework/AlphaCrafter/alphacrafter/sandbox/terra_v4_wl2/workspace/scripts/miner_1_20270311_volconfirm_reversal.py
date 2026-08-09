import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date') for a in A}
# Volume-confirmed reversal: prior 3d loss is stronger when volume is elevated vs its 20d median.
F={a: -(D[a].close.pct_change(3)) * np.log1p(D[a].volume/(D[a].volume.rolling(20).median()+1e-12)) for a in A}
rows=[]; sig=[]
for d in sorted(set().union(*[set(x.index) for x in D.values()])):
 vals={a:F[a].get(d,np.nan) for a in A}; good=[v for v in vals.values() if np.isfinite(v)]
 if len(good)<8: continue
 med=np.nanmedian(good)
 for a in A:
  if np.isfinite(vals[a]): sig.append((d,a,vals[a]-med))
 for h in [1,5,10]:
  f=[];y=[]
  for a in A:
   if d not in D[a].index or not np.isfinite(vals[a]): continue
   i=D[a].index.get_loc(d)
   if i+h<len(D[a]): f.append(vals[a]-med); y.append(D[a].close.iloc[i+h]/D[a].close.iloc[i]-1)
  if len(f)>=8: rows.append((d,h,spearmanr(f,y).statistic,len(f)))
df=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 x=df[df.h==h]; print('H',h,'dates',len(x),'avg_n',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,4),'IC',round(x.ic.mean(),6),'ICIR',round(x.ic.mean()/x.ic.std(),6),'hit',round((x.ic>0).mean(),4))
 for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-06-30'),('2026-07','2027')]:
  z=x.set_index('date').loc[lo:hi].ic; print(lo,len(z),round(z.mean(),6),round(z.mean()/z.std(),6) if len(z)>1 else np.nan)
w=pd.DataFrame(sig,columns=['date','asset','signal']).pivot(index='date',columns='asset',values='signal'); print('turnover',round(w.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
pd.DataFrame(sig,columns=['date','asset','signal']).to_csv('../persistent/factor_signals_miner_1_20270311_volconfirm_reversal.csv',index=False)
