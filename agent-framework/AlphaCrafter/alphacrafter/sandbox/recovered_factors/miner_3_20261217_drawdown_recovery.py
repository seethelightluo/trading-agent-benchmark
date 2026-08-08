import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end='2026-12-17'
S=[]
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.rename(a);S.append(d)
P=pd.concat(S,axis=1,join='outer').sort_index().loc[:end].ffill(); R=P.pct_change()
# Drawdown recovery: rebound from 60d rolling low, scaled by recent volatility.
low=P.rolling(60,min_periods=40).min(); rebound=P/low-1
vol=R.rolling(20,min_periods=15).std(); F=rebound/vol.replace(0,np.nan)
print('candidate drawdown_recovery_60d; dates',len(R),'assets',len(A))
for h in [1,5,10,20]:
 z=[];ns=[]
 for i in range(len(R)-h):
  q=pd.concat([F.iloc[i],R.iloc[i+1:i+1+h].sum()],axis=1).dropna()
  if len(q)>=8:
   z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 z=np.array(z); print('horizon',h,'dates',len(z),'avg_n',round(np.mean(ns),2),'IC',round(np.mean(z),6),'ICIR',round(np.mean(z)/np.std(z,ddof=1),6),'hit',round(np.mean(z>0),4))
print('coverage',round(F.notna().stack().mean(),4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(1).mean(),4))
# compare against currently admitted signal forms, pooled Spearman
lib={'trend20':R.rolling(20,min_periods=15).sum()/R.rolling(20,min_periods=15).std(),'vol20':R.rolling(20,min_periods=15).std(),'shortrev5':-R.rolling(5,min_periods=4).sum()/R.rolling(5,min_periods=4).std(),'ravmom':R.rolling(20,min_periods=15).sum()/R.rolling(20,min_periods=15).std()}
for n,x in lib.items():
 q=pd.concat([F.stack(),x.stack()],axis=1).dropna();print('rho',n,round(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic,6),'n',len(q))
# yearly IC robustness at 1d
x=[]
for i in range(len(R)-1):
 q=pd.concat([F.iloc[i],R.iloc[i+1]],axis=1).dropna()
 if len(q)>=8:x.append((R.index[i],spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
df=pd.DataFrame(x,columns=['date','ic']).set_index('date');print('yearly',df.groupby(df.index.year).ic.agg(['mean','count']).round(5).to_dict('index'))
