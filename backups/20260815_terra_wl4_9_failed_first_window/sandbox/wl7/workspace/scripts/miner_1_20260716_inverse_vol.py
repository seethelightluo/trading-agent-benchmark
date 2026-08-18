import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; d={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv');x.date=pd.to_datetime(x.date);x=x[x.date<='2026-07-15'];x=x.set_index('date').sort_index();r=x.close.pct_change()
 # inverse volatility: stable assets preferred, using only past returns
 d[s]=pd.DataFrame({'f':-r.rolling(20,min_periods=15).std(),'r':r})
F=pd.concat({s:v.f for s,v in d.items()},axis=1); n=[]
for h in [1,5,10]:
 R=pd.concat({s:d[s].r.rolling(h).sum().shift(-h) for s in U},axis=1); ic=[];ns=[];ds=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],R.loc[dt]],axis=1).dropna()
  if len(z)>=8:ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(dt)
 a=np.array(ic);print('h',h,'dates',len(a),'meanN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'cov',F.notna().sum(axis=1).mean()/15)
 if h==1:
  for y in sorted(set(x.year for x in ds)):
   q=a[[z.year==y for z in ds]];print(y,round(q.mean(),4))
print('turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
