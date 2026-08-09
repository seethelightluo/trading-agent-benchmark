import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17')
P=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill().loc[:cut]; R=P.pct_change()
def z(x):
 m=x.mean(axis=1); sd=x.std(axis=1).replace(0,np.nan); return x.sub(m,axis=0).div(sd,axis=0)
def ev(name,F):
 for h in [1,5,10]:
  Y=P.pct_change(h).shift(-h); a=[]; ns=[]
  for d in F.index:
   q=pd.concat([F.loc[d].rename('f'),Y.loc[d].rename('y')],axis=1).dropna()
   if len(q)>=8: a.append(q.f.corr(q.y,method='spearman'));ns.append(len(q))
  a=pd.Series(a).dropna();print(name,h,len(a),round(np.mean(ns),2),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6),round((a>0).mean(),4))
for w in [.25,.5,.75,1.0]:
 F=z(-R.rolling(1).sum())+w*z(-R.rolling(3).sum()); ev('reversal1+%g reversal3'%w,F)
F=z(-R.rolling(3).sum())/z(R.rolling(20).std())
ev('volscaled3',F)
print('period',P.index.min().date(),P.index.max().date(),'assets',len(U))
