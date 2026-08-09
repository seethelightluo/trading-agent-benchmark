import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
R=pd.concat([P[a].pct_change().rename(a) for a in A],axis=1); base=-R.rolling(3,min_periods=3).sum().shift(1)
disp=R.shift(1).std(axis=1).rolling(5,min_periods=3).mean()
rows=[];sig=[]
for dt in sorted(set().union(*[set(x.index) for x in P.values()])):
 vals={a:base[a].get(dt,np.nan)*(1+10*disp.get(dt,0)) for a in A}
 good=[v for v in vals.values() if np.isfinite(v)]
 if len(good)<8:continue
 med=np.median(good)
 for a in A:sig.append((dt,a,vals[a]-med if np.isfinite(vals[a]) else np.nan))
 for h in [1,3,5,10]:
  f=[];y=[]
  for a in A:
   if dt not in P[a].index:continue
   i=P[a].index.get_loc(dt);z=vals[a]-med
   if np.isfinite(z) and i+h<len(P[a]):f.append(z);y.append(P[a].iloc[i+h]/P[a].iloc[i]-1)
  if len(f)>=8:
   q=spearmanr(f,y).statistic
   if np.isfinite(q):rows.append((dt,h,q,len(f)))
d=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,3,5,10]:
 x=d[d.h==h];print('H',h,'dates',len(x),'avg_n',round(x.n.mean(),2),'IC',round(x.ic.mean(),6),'ICIR',round(x.ic.mean()/x.ic.std(),6),'hit',round((x.ic>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  z=x.set_index('date').loc[lo:hi].ic;print('REG',lo,len(z),round(z.mean(),6))
w=pd.DataFrame(sig,columns=['date','asset','signal']).pivot(index='date',columns='asset',values='signal');print('turnover',round(w.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6));pd.DataFrame(sig,columns=['date','asset','signal']).to_csv('../persistent/factor_signals_miner_1_20270225_dispersion_scaled_reversal.csv',index=False)
