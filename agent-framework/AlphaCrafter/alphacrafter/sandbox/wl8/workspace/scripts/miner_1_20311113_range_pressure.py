import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2031-11-13'); fs={}; ps={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').loc[:end]
 # lagged multi-day close location / range, blended with 5d return reversal
 loc=((d.close-d.low)/(d.high-d.low).replace(0,np.nan)).rolling(3).mean().shift(1)
 rev=(-np.log(d.close/d.close.shift(5))).shift(1)
 fs[a]=(rev*(0.5+loc)).rename(a); ps[a]=d.close.rename(a)
F=pd.concat(fs,axis=1); P=pd.concat(ps,axis=1); R=np.log(P.shift(-10)/P); vals=[]
for dt in F.index:
 z=pd.concat([F.loc[dt],R.loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
a=np.array(vals); print('dates',len(a),'avg_names',round(F.notna().sum(1).mean(),2),'coverage',round(F.notna().mean().mean(),4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for n in [180,365,756]:
 b=a[-n:];print('recent',n,round(b.mean(),6),round(b.mean()/b.std(ddof=1),6),len(b))
