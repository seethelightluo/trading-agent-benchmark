import numpy as np,pandas as pd
from pathlib import Path
root=Path('../persistent'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def L(s,m=False):
 p=root/('index_data' if m else 'stock_data')/(s+'.csv'); d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); return d.close.astype(float)
p=pd.concat({s:L(s) for s in syms},axis=1,join='inner'); d=L('DXY',True); p=p.join(d.rename('DXY'),how='inner').loc[:'2026-09-09']; d=p.pop('DXY')
r=p.pct_change(); dr=d.pct_change(); mu=dr.rolling(60,min_periods=30).mean(); sd=dr.rolling(60,min_periods=30).std(); shock=(dr-mu)/sd
f=r.mul(-np.sign(shock)*np.minimum(abs(shock),2.0),axis=0)
for h in [1,5,10]:
 yy=p.shift(-h).div(p)-1; a=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]));ns.append(len(z))
 q=pd.Series(a);print('h',h,'dates',len(q),'names',np.mean(ns),'coverage',np.sum(ns)/(len(ns)*15),'IC',q.mean(),'ICIR',q.mean()/q.std(),'hit',(q>0).mean())
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-09-09')]:
 q=[]
 for dt in f.loc[a:b].index:
  z=pd.concat([f.loc[dt],(p.shift(-1).div(p)-1).loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1]))
 q=pd.Series(q);print(a,'n',len(q),'ICIR',q.mean()/q.std(),'IC',q.mean())
print('end',f.index.max().date())
