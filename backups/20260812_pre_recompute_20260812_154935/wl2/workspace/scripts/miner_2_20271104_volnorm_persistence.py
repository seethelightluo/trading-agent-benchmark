import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s,idx):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']);d.date=d.date.dt.normalize();return d.set_index('date').close.reindex(idx)
b=pd.read_csv('../persistent/stock_data/SPX.csv',parse_dates=['date']);idx=pd.DatetimeIndex(b.date.dt.normalize().unique());idx=idx[(idx>='2020-04-01')&(idx<='2027-11-03')]
C=pd.DataFrame({s:load(s,idx) for s in U});R=C.pct_change();v=R.rolling(20,min_periods=15).std();mom=R.rolling(10,min_periods=8).sum();F=(mom/v).shift(1);F=F.sub(F.median(axis=1),axis=0)
print('dates',len(idx),'N',len(U))
for h in [1,3,5,10]:
 y=C.shift(-h)/C-1;z=[];ns=[]
 for d in idx:
  q=pd.concat([F.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.f,q.y).statistic);ns.append(len(q))
 z=np.asarray(z);print(h,len(z),np.mean(ns),z.mean(),z.mean()/z.std(ddof=1),np.mean(z>0))
print('coverage',F.notna().sum().sum()/F.size,'turnover',F.rank(pct=True).diff().abs().mean().mean())
