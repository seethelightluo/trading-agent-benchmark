import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end='2026-12-17'
Cs=[];Vs=[]
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end]
 Cs.append(d.close.rename(a)); Vs.append(d.volume.rename(a))
P=pd.concat(Cs,axis=1,join='outer').sort_index().ffill(); V=pd.concat(Vs,axis=1,join='outer').sort_index().ffill(); R=P.pct_change()
# Volume-confirmed medium-term pressure: return over 5 sessions multiplied by abnormal volume, normalized by 20d vol.
rv=V/V.rolling(20,min_periods=15).mean(); F=R.rolling(5,min_periods=5).sum()*rv.rolling(5,min_periods=5).mean()/R.rolling(20,min_periods=15).std().replace(0,np.nan)
print('candidate volume_confirmed_pressure_5d; dates',len(R),'assets',len(A))
for h in [1,5,10,20]:
 z=[];ns=[]
 for i in range(len(R)-h):
  q=pd.concat([F.iloc[i],R.iloc[i+1:i+1+h].sum()],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 z=np.array(z);print('horizon',h,'dates',len(z),'avg_n',round(np.mean(ns),2),'IC',round(np.mean(z),6),'ICIR',round(np.mean(z)/np.std(z,ddof=1),6),'hit',round(np.mean(z>0),4))
print('coverage',round(F.notna().stack().mean(),4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(1).mean(),4))
lib={'trend20':R.rolling(20,min_periods=15).sum()/R.rolling(20,min_periods=15).std(),'vol20':R.rolling(20,min_periods=15).std(),'shortrev5':-R.rolling(5,min_periods=4).sum()/R.rolling(5,min_periods=4).std(),'relvol20':rv}
for n,x in lib.items():
 q=pd.concat([F.stack(),x.stack()],axis=1).dropna();print('rho',n,round(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic,6),'n',len(q))
