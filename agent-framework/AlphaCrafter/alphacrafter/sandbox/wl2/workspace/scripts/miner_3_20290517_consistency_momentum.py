import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for a in A:
 p='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d.date=pd.to_datetime(d.date); d=d[d.date<='2029-05-16'].set_index('date').sort_index(); D[a]=d.close
px=pd.concat(D,axis=1).sort_index(); r=px.pct_change()
mom=r.rolling(20,min_periods=20).sum(); consistency=(r>0).rolling(20,min_periods=20).mean(); vol=r.rolling(20,min_periods=20).std()
f=-mom*consistency/vol
lagged=f.shift(1)
for h in [1,3,5,10]:
 fr=px.pct_change(h).shift(-h); vals=[]; ns=[]; turns=[]
 for i,dt in enumerate(lagged.index):
  z=pd.concat([lagged.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
   if i:
    q=pd.concat([lagged.iloc[i],lagged.iloc[i-1]],axis=1).dropna(); turns.append(np.mean((q.iloc[:,0].rank(pct=True)-q.iloc[:,1].rank(pct=True)).abs()))
 x=np.array(vals); print('h',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4),'turn',round(np.nanmean(turns),4))
fr=px.pct_change(1).shift(-1); vals=[]
for dt in lagged.index:
 z=pd.concat([lagged.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
for name,sel in [('pre2027',lambda d:d<pd.Timestamp('2027-01-01')),('2027-28',lambda d:pd.Timestamp('2027-01-01')<=d<pd.Timestamp('2029-01-01')),('2029+',lambda d:d>=pd.Timestamp('2029-01-01'))]:
 x=np.array([v for d,v in vals if sel(d)]); print(name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
print('instruments',len(D),'rows',len(px))
lagged.to_csv('scripts/miner_3_20290517_consistency_momentum_signal.csv')
