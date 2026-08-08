import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat([pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.rename(a) for a in A],axis=1,join='inner').sort_index().loc[:'2027-02-24']
R=P.pct_change(fill_method=None)
# Cross-sectional rank acceleration: recent 5-day rank relative to 20-day rank.
f=R.rolling(5,min_periods=5).sum().rank(axis=1,pct=True)-R.rolling(20,min_periods=15).sum().rank(axis=1,pct=True)
print('Rank acceleration | dates',len(P),'assets',len(A),'coverage',round(f.notna().stack().mean(),4))
for h in [1,5,10,20]:
 z=[]; ns=[]; dates=[]
 for i in range(len(R)-h):
  q=pd.concat([f.iloc[i],R.iloc[i+1:i+1+h].sum()],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q));dates.append(R.index[i])
 z=np.asarray(z);print('H',h,'dates',len(z),'mean_n',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round(np.mean(z>0),4))
 if h==1:
  d=pd.DataFrame({'date':dates,'ic':z});print('year',d.groupby(pd.to_datetime(d.date).dt.year).ic.agg(['mean','count']).round(5).to_dict('index'))
print('rank turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
# library proxies and all persisted factor signal approximations
libs={'trend':R.rolling(20,min_periods=15).sum()/R.rolling(20,min_periods=15).std(),'vol':R.rolling(20,min_periods=15).std(),'shortrev':-R.rolling(5,min_periods=4).sum()/R.rolling(5,min_periods=4).std(),'vixrev':-R.rolling(3,min_periods=3).sum(),'rav':R.rolling(20,min_periods=15).sum()/R.rolling(20,min_periods=15).std(),'peer':R.rolling(2).mean().sub(R.rolling(2).mean().mean(axis=1),axis=0)}
mx=0
for n,x in libs.items():
 q=pd.concat([f.stack(),x.stack()],axis=1).dropna();rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic;mx=max(mx,abs(rho));print('rho',n,round(rho,6))
print('MAX_PROXY',round(mx,6))
