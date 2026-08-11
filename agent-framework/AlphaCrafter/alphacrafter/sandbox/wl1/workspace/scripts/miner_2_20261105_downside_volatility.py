import os,pandas as pd,numpy as np
from scipy.stats import spearmanr
cut=pd.Timestamp('2026-07-15'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv'); d.date=pd.to_datetime(d.date); px[s]=d[d.date<=cut].set_index('date').close.astype(float)
P=pd.DataFrame(px); F=pd.DataFrame(index=P.index)
for s in syms:
 r=px[s].pct_change(); F[s]=-r.where(r<0).rolling(30,min_periods=20).std().shift(1)
for h in [5,10,20]:
 out=[];ns=[];ds=[]
 for i,dt in enumerate(P.index[:-h]):
  z=pd.concat([F.loc[dt],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(dt)
 a=np.array(out); print('H',h,'valid_dates',len(a),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
 if h==10:
  for y in sorted(set(d.year for d in ds)):
   q=a[[d.year==y for d in ds]]; print('year',y,'IC',q.mean(),'n',len(q))
print('turnover',F.rank(axis=1).diff().abs().mean(axis=1).mean()/14)
