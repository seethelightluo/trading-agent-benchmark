import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat([pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.rename(a) for a in A],axis=1,join='outer').sort_index().loc[:'2027-01-13'].ffill();R=P.pct_change()
low=P.rolling(60,min_periods=40).min(); F=(P/low-1)/R.rolling(20,min_periods=15).std()
print('drawdown recovery 60d; dates',len(R),'assets',len(A))
for h in [1,5,10,20]:
 z=[];ns=[]
 for i in range(len(R)-h):
  q=pd.concat([F.iloc[i],R.iloc[i+1:i+1+h].sum()],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 z=np.array(z);print(h,'dates',len(z),'mean_n',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round(np.mean(z>0),4))
print('coverage',round(F.notna().stack().mean(),4),'turn',round(F.rank(axis=1,pct=True).diff().abs().mean(1).mean(),4))
libs={'trend':R.rolling(20,min_periods=15).sum()/R.rolling(20,min_periods=15).std(),'vol':R.rolling(20,min_periods=15).std(),'shortrev':-R.rolling(5,min_periods=4).sum()/R.rolling(5,min_periods=4).std(),'ravmom':R.rolling(20,min_periods=15).sum()/R.rolling(20,min_periods=15).std()}
for n,x in libs.items():
 q=pd.concat([F.stack(),x.stack()],axis=1).dropna();print('rho',n,round(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic,6),len(q))
