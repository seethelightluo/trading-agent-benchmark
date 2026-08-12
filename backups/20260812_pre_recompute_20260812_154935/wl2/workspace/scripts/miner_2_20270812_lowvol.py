import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}; p=pd.DataFrame(p).sort_index(); p=p[p.index<=pd.Timestamp('2027-08-11')]
r=p.pct_change(); f=-r.rolling(20).std();
for h in [1,3,5,10]:
 y=p.shift(-h)/p-1;q=[];ds=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(d);ns.append(len(z))
 q=np.array(q); print('h',h,'dates',len(q),'avgN',np.mean(ns),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',np.mean(q>0))
 if h==1:
  for a,b in [('2020','2021'),('2022','2023'),('2024','2025'),('2026','2027')]:
   x=np.array([v for d,v in zip(ds,q) if a<=str(d.year)<=b]); print(a,b,len(x),x.mean(),x.mean()/x.std(ddof=1))
print('coverage',f.notna().mean().mean(),'turnover',((f.rank(pct=True)-f.shift().rank(pct=True)).abs().mean(axis=1)).mean())
