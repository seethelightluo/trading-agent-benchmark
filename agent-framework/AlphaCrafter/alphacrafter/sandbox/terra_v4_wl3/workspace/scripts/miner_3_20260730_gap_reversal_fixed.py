import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:end] for s in U}
C=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index(); F=pd.DataFrame(index=C.index,columns=U,dtype=float)
for s,d in D.items():
 q=(d.open/d.close.shift(1)-1).replace([np.inf,-np.inf],np.nan); a=-q.rolling(3,min_periods=3).mean(); F[s]=a.reindex(F.index)
Y={h:C.pct_change(h).shift(-h) for h in [1,5,10]}
for h in [1,5,10]:
 z=[]; ns=[]; ds=[]
 for dt in F.index:
  a=pd.concat([F.loc[dt],Y[h].loc[dt]],axis=1).dropna()
  if len(a)>=8 and a.iloc[:,0].nunique()>1 and a.iloc[:,1].nunique()>1:z.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman'));ns.append(len(a));ds.append(dt)
 z=np.array(z); ix=pd.DatetimeIndex(ds); print('H',h,'dates',len(z),'meanN',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
 if h==1:
  print('coverage',round(F.notna().mean().mean(),4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
  for n,mask in [('2020-22',ix<='2022-12-31'),('2023-24',(ix>='2023-01-01')&(ix<='2024-12-31')),('2025-26',ix>='2025-01-01')]:
   q=z[mask];print(n,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
F.to_csv('scripts/miner_3_20260730_gap_reversal_signal.csv'); print('artifact dates',F.index.min(),F.index.max(),'assets',len(U))
