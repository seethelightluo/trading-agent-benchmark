import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end='2026-07-15'
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').close for s in U}
p=pd.concat(P,axis=1,sort=True).loc[:end]; r=np.log(p).diff(); b=r.mean(axis=1); w=30
mb=b.rolling(w,min_periods=20).mean(); mb2=(b*b).rolling(w,min_periods=20).mean(); var=mb2-mb*mb
F=pd.DataFrame(index=p.index,columns=U,dtype=float)
for s in U:
 x=r[s]; mx=x.rolling(w,min_periods=20).mean(); mxb=(x*b).rolling(w,min_periods=20).mean(); cov=mxb-mx*mb; beta=cov/var
 F[s]=(x-beta*b).rolling(10,min_periods=8).sum()
for h in [1,5,10]:
 y=p.pct_change(h).shift(-h); z=[]; ns=[]; ds=[]
 for dt in F.index:
  a=pd.concat([F.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(a)>=8 and a.iloc[:,0].nunique()>1 and a.iloc[:,1].nunique()>1:z.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman'));ns.append(len(a));ds.append(dt)
 z=np.array(z); ix=pd.DatetimeIndex(ds);print('h',h,'dates',len(z),'meanN',np.mean(ns),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
 if h==1:
  print('coverage',F.notna().mean().mean(),'turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
  for lab,m in [('2020-22',ix<='2022-12-31'),('2023-24',(ix>='2023-01-01')&(ix<='2024-12-31')),('2025-26',ix>='2025-01-01')]:
   q=z[m];print(lab,len(q),q.mean(),q.mean()/q.std(ddof=1))
F.to_csv('scripts/miner_1_20260827_residual_momentum_signal.csv');print('artifact scripts/miner_1_20260827_residual_momentum_signal.csv')
