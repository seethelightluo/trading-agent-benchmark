import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
# Consistency-weighted medium-term reversal: reverse 30d return, emphasize
# persistent directional moves, volatility-normalized and lagged one session.
m=p/p.shift(30)-1; cons=(r>0).rolling(30).mean(); v=r.rolling(40).std()*np.sqrt(20)
f=(-m*cons/v).shift(1)
print('period',p.index.min(),p.index.max(),'assets',len(p.columns),'rows',len(p))
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; q=[]; ns=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 q=pd.Series(q).dropna(); print('horizon',h,'ic',q.mean(),'icir',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'dates',len(q),'avg_n',np.mean(ns))
rank=f.rank(axis=1,pct=True); print('coverage',f.notna().mean().mean(),'turnover',rank.diff().abs().mean(axis=1).dropna().mean())
for n in [180,500,750]:
 vals=[]
 for dt in p.index[-n:]:
  z=pd.concat([f.loc[dt],(p.shift(-20)/p-1).loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(vals).dropna(); print('recent20',n,'ic',q.mean(),'icir',q.mean()/q.std(ddof=1),'dates',len(q))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20340501_consistency_reversal_signal.csv',index=False)
