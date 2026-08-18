import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; p={}
for a in A:p[a]=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close']
pd0=pd.DataFrame(p).sort_index().loc[:'2028-12-13']; r=pd0.pct_change(); f=(-(pd0.pct_change(5))/(r.rolling(20).std()*np.sqrt(5))).clip(-8,8)
print('assets',len(A),'dates',len(pd0),'coverage',round(f.notna().mean().mean(),4),'valid',sum(f.notna().sum(1)>=8))
for h in [1,5,10,20]:
 s=[]; ns=[]
 for i in range(len(pd0)-h):
  z=pd.concat([f.iloc[i],pd0.pct_change(h).iloc[i+h]],axis=1).dropna()
  if len(z)>=8:s.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 s=pd.Series(s); q=s.tail(250); print('h',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC',round(s.mean(),5),'ICIR',round(s.mean()/s.std(),5),'hit',round((s>0).mean(),4),'recent',round(q.mean(),5),round(q.mean()/q.std(),5))
