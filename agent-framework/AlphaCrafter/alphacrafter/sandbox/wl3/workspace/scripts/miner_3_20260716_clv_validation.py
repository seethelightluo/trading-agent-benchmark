import pandas as pd, numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15'); base=Path('../persistent/stock_data')
O={}
for s in U:
 d=pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date').sort_index(); d=d[d.index<=end]
 rng=(d.high-d.low).replace(0,np.nan); O[s]=pd.DataFrame({'f':-(2*(d.close-d.low)/rng-1),'p':d.close})
P=pd.concat({s:x.p for s,x in O.items()},axis=1).sort_index().ffill(); F=pd.concat({s:x.f for s,x in O.items()},axis=1).reindex(P.index); R=P.pct_change()
print('dates',len(P),'instruments',len(U))
def evalh(h, sub=None):
 y=P.shift(-h)/P-1; q=[]; ns=[]
 ix=F.index if sub is None else F.loc[sub].index
 for dt in ix:
  z=pd.concat([F.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 q=np.array(q); return len(q),np.mean(ns),q.mean(),q.mean()/(q.std(ddof=1)+1e-12),(q>0).mean()
for h in [1,5,10,20]: print('h',h,evalh(h))
rank=F.rank(axis=1,pct=True); print('coverage',F.notna().sum(axis=1).ge(8).mean(),'turnover',((rank-rank.shift()).abs().mean(axis=1)).dropna().mean())
for label,sl in [('2020-2022',slice('2020','2022-12-31')),('2023-2024',slice('2023','2024-12-31')),('2025-2026',slice('2025','2026-07-15'))]: print(label,evalh(1,sl))
# compare flattened factor to existing close-return factors
for n,w in [('mom20',20),('rev5',5),('riskmom20',20)]:
 if n=='riskmom20': X=R.rolling(20).sum()/(R.rolling(60).std()*np.sqrt(20))
 elif n=='mom20': X=P.pct_change(20)
 else: X=-P.pct_change(5)
 z=pd.concat([F.stack(),X.stack()],axis=1).dropna(); print('corr',n,z.iloc[:,0].corr(z.iloc[:,1]))
