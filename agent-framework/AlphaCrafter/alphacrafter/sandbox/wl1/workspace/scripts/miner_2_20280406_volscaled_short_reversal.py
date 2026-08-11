import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2028-04-05')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut] for s in U}
idx=sorted(set().union(*[set(x.index) for x in P.values()]));C=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill();R=C.pct_change()
# Volatility-scaled short-term reversal: negative 10-session return, normalized by trailing 20-session volatility; lagged one session.
ret=R.rolling(10,min_periods=8).sum();vol=R.rolling(20,min_periods=15).std()*np.sqrt(10);f=(-ret/(vol+0.004)).shift(1)
print('factor volscaled_short_reversal universe',len(U),'dates',len(C),'cutoff',C.index.max().date())
for h in [5,10,20]:
 a=[];ns=[];ds=[]
 for i in range(len(C)-h):
  q=pd.concat([f.iloc[i].rename('f'),(C.iloc[i+h]/C.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:a.append(spearmanr(q.f,q.y).statistic);ns.append(len(q));ds.append(C.index[i])
 a=np.array(a);ds=pd.DatetimeIndex(ds);print('h',h,'valid_dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 for lab,st in [('2024+','2024-01-01'),('2025+','2025-01-01'),('2026+','2026-01-01'),('2027+','2027-01-01'),('2028+','2028-01-01')]:
  z=a[ds>=pd.Timestamp(st)];print(lab,'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None,'dates',len(z))
rk=f.rank(axis=1,pct=True);print('turnover',round((rk-rk.shift()).abs().stack().groupby(level=0).mean().dropna().mean(),6),'coverage_dates',int(f.notna().sum(axis=1).ge(8).sum()));f.to_csv('scripts/miner_2_20280406_volscaled_short_reversal_signal.csv',index_label='date')
