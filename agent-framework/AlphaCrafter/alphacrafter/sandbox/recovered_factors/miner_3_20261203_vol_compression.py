import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end='2026-12-03'
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in A}).sort_index().loc[:end]
R=P.pct_change(); f=R.rolling(10,min_periods=8).std()/R.rolling(60,min_periods=40).std()
print('candidate volatility compression ratio 10d/60d; dates',len(R),'assets',len(A))
for h in [1,5,10,20]:
 z=[];ns=[]
 for i in range(len(R)-h):
  q=pd.concat([f.iloc[i],R.iloc[i+1:i+1+h].sum()],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 z=np.asarray(z);print('h',h,'dates',len(z),'mean_n',round(np.mean(ns),2),'IC',round(np.mean(z),6),'ICIR',round(np.mean(z)/np.std(z,ddof=1),6),'hit',round(np.mean(z>0),4))
print('coverage',round(f.notna().stack().mean(),4),'turn',round(f.rank(axis=1,pct=True).diff().abs().mean(1).mean(),4))
lib={'trend':R.rolling(20,min_periods=15).sum()/R.rolling(20,min_periods=15).std(),'vol':R.rolling(20,min_periods=15).std(),'shortrev':-R.rolling(5,min_periods=4).sum()/R.rolling(5,min_periods=4).std()}
for n,x in lib.items():
 q=pd.concat([f.stack(),x.stack()],axis=1).dropna();print('rho',n,round(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic,6),len(q))
