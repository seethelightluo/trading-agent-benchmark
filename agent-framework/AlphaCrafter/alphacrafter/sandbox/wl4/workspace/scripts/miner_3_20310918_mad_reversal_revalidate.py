import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2031-09-17'); p={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close; p[s]=d[d.index<=cutoff]
px=pd.DataFrame(p).sort_index(); r=px.pct_change(); res=r.rolling(15,min_periods=10).sum(); res=res.sub(res.median(axis=1),axis=0)
am=pd.DataFrame(index=px.index,columns=px.columns)
for c in r: am[c]=r[c].rolling(40,min_periods=20).apply(lambda x: np.median(np.abs(x-np.median(x))),raw=True)*1.4826
csdisp=r.rolling(15,min_periods=10).std().median(axis=1); threshold=csdisp.rolling(120,min_periods=60).median(); gate=(csdisp>threshold).astype(float)
f=(-res/(am+1e-8)*gate.values[:,None]).shift(1)
fr=px.shift(-10)/px-1; vals=[]; ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
x=pd.Series(vals); print('H 10 dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for n in [365,730]:
 vals=[]
 for dt in f.index[-n:]:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=pd.Series(vals); print('recent',n,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
print('coverage',round(f.notna().sum().sum()/px.notna().sum().sum(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'dates',len(px),'instruments',len(U))
