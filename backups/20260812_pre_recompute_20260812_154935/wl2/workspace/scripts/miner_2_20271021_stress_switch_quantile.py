import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def L(s,idx):
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']);x.date=x.date.dt.normalize();return x.set_index('date').close.reindex(idx)
sp=pd.read_csv('../persistent/stock_data/SPX.csv',parse_dates=['date']);idx=sp.date.dt.normalize();idx=idx[(idx>='2020-04-01')&(idx<='2027-10-20')]
C=pd.DataFrame({s:L(s,idx) for s in U});R=C.pct_change(); v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']);v.date=v.date.dt.normalize();v=v.set_index('date').close.reindex(idx).shift(1)
q=v.rolling(120,min_periods=80).quantile(.75); stress=(v>q).astype(float)
r5=R.rolling(5,min_periods=5).sum();r20=R.rolling(20,min_periods=20).sum();vol=R.rolling(20,min_periods=20).std(); a=-r5/vol;a=a.sub(a.median(axis=1),axis=0);b=r20/vol;b=b.sub(b.median(axis=1),axis=0);F=(a.mul(stress,axis=0)+b.mul(1-stress,axis=0)).shift(1)
for h in [1,3,5,10]:
 y=C.shift(-h)/C-1; z=[];ns=[]
 for d in idx:
  x=pd.concat([F.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(x)>=8:z.append(spearmanr(x.f,x.y).statistic);ns.append(len(x))
 z=np.array(z);print(h,len(z),np.mean(ns),z.mean(),z.mean()/z.std(ddof=1),np.mean(z>0))
print('coverage',F.notna().sum().sum()/F.size,'turnover',F.rank(pct=True).diff().abs().mean().mean(),'stress',stress.mean())
