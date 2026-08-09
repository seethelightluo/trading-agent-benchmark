import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat([pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.rename(a) for a in A],axis=1,join='inner').sort_index().loc[:'2026-12-31']; R=P.pct_change()
# Cross-asset relative reversal: recent return minus its contemporaneous cross-sectional median, smoothed 3d and risk scaled
rel=R.sub(R.median(axis=1),axis=0)
f=-rel.rolling(3,min_periods=3).sum()/R.rolling(20,min_periods=15).std()
print('candidate cross-asset relative reversal; dates',len(R),'assets',len(A))
for h in [1,5,10,20]:
 z=[]; ns=[]
 for i in range(len(R)-h):
  q=pd.concat([f.iloc[i],R.iloc[i+1:i+1+h].sum()],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 z=np.array(z); print(h,'dates',len(z),'mean_n',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round(np.mean(z>0),4))
print('coverage',round(f.notna().stack().mean(),4),'turn',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
libs={'trend':R.rolling(20,min_periods=15).sum()/R.rolling(20,min_periods=15).std(),'vol':R.rolling(20,min_periods=15).std(),'shortrev':-R.rolling(5,min_periods=4).sum()/R.rolling(5,min_periods=4).std()}
for n,x in libs.items():
 q=pd.concat([f.stack(),x.stack()],axis=1).dropna(); print('rho',n,round(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic,6),len(q))
