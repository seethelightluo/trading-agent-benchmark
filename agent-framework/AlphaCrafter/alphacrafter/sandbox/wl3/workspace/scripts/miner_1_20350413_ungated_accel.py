import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
S={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date); S[s]=x.drop_duplicates('date').set_index('date').close.astype(float)
p=pd.concat(S,axis=1).sort_index(); r=p.pct_change(); v=r.rolling(20,min_periods=15).std(); f=(p.pct_change(20)-p.pct_change(60)/3)/(v+1e-12); f=f.replace([np.inf,-np.inf],np.nan)
print('candidate ungated trend acceleration / 20d vol')
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1; ic=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8: ic.append(z.iloc[:,0].corr(z.iloc[:,1])); ns.append(len(z))
 q=pd.Series(ic); print('horizon',h,'dates',len(q),'avg_names',np.mean(ns),'coverage',np.mean(ns)/15,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',np.mean(q>0),'recent252',q.tail(252).mean(),q.tail(252).mean()/q.tail(252).std(ddof=1))
# 4 blocks at 10d
fr=p.shift(-10)/p-1; q=[]
for d in f.index:
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8:q.append((d,z.iloc[:,0].corr(z.iloc[:,1])))
q=pd.Series(dict(q));
for i,b in enumerate(np.array_split(q,4),1): print('block',i,'IC',b.mean(),'ICIR',b.mean()/b.std(ddof=1))
print('rank_turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
