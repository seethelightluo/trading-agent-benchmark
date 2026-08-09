import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
R=pd.concat([P[a].pct_change().rename(a) for a in A],axis=1)
ret3=R.rolling(3,min_periods=3).sum().shift(1)
market=R.mean(axis=1)
# lagged rolling beta to common cross-asset market, then residual 3d return; inverse vol scaling
vol=R.rolling(20,min_periods=10).std().shift(1)
beta=pd.DataFrame(index=R.index,columns=A,dtype=float)
for a in A:
 beta[a]=R[a].rolling(60,min_periods=30).cov(market)/market.rolling(60,min_periods=30).var()
resid=ret3-beta.mul(market.rolling(3).sum().shift(1),axis=0)
sig=-resid/vol
rows=[];out=[]
for dt in sorted(set().union(*[set(x.index) for x in P.values()])):
 vals={a:sig[a].get(dt,np.nan) for a in A}; good=np.array([v for v in vals.values() if np.isfinite(v)])
 if len(good)<8: continue
 med=np.nanmedian(good)
 for a in A:
  v=vals[a]; out.append((dt,a,v-med if np.isfinite(v) else np.nan))
 for h in [1,3,5,10]:
  f=[];y=[]
  for a in A:
   if dt not in P[a].index: continue
   i=P[a].index.get_loc(dt); v=vals[a]-med
   if np.isfinite(v) and i+h<len(P[a]):f.append(v);y.append(P[a].iloc[i+h]/P[a].iloc[i]-1)
  if len(f)>=8:
   q=spearmanr(f,y).statistic
   if np.isfinite(q):rows.append((dt,h,q,len(f)))
d=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,3,5,10]:
 x=d[d.h==h]; print('H',h,'dates',len(x),'avg_n',round(x.n.mean(),2),'IC',round(x.ic.mean(),6),'ICIR',round(x.ic.mean()/x.ic.std(),6),'hit',round((x.ic>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  z=x.set_index('date').loc[lo:hi].ic; print('REG',lo,len(z),round(z.mean(),6),round(z.mean()/z.std(),6) if len(z)>1 else np.nan)
pd.DataFrame(out,columns=['date','asset','signal']).to_csv('../persistent/factor_signals_miner_1_20270227_residual_vol_reversal3.csv',index=False)
