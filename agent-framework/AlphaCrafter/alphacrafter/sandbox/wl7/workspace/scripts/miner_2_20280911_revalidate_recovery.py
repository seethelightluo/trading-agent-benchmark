import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];end=pd.Timestamp('2028-09-11');P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']);P[s]=d[d.date<=end].set_index('date').close
px=pd.DataFrame(P).sort_index().ffill();r=px.pct_change();down=r.where(r<0).rolling(40,min_periods=20).std();mom=px.shift(5)/px.shift(25)-1;trend=px.shift(5)/px.shift(65)-1
sig=(mom/(down*np.sqrt(252)+.01)*np.where(trend>0,1,.25)).shift(1);f=px.shift(-10)/px-1;z=[];ds=[];ns=[]
for dt in sig.index:
 a=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
 if len(a)>=8:
  q=spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic
  if np.isfinite(q):z.append(q);ds.append(dt);ns.append(len(a))
z=np.array(z);print('dates',len(z),'avgN',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4),'coverage',round(sig.notna().sum().sum()/sig.size,4),'turn',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
for lab,a,b in [('2020-22','2020-01-01','2022-12-31'),('2023-25','2023-01-01','2025-12-31'),('2026-28','2026-01-01','2028-09-11')]:
 q=z[[a<=str(d.date())<=b for d in ds]];print(lab,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
