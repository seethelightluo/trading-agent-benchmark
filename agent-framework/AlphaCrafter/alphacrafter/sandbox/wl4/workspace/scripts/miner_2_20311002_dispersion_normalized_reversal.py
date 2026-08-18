import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2031-10-01'); p={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close
 p[s]=d[d.index<=cutoff]
px=pd.DataFrame(p).sort_index(); r=px.pct_change(); ret=r.rolling(10,min_periods=8).sum(); resid=ret.sub(ret.median(axis=1),axis=0)
disp=r.rolling(20,min_periods=12).std().median(axis=1); disp_med=disp.rolling(60,min_periods=30).median(); scale=(disp/(disp_med+1e-8)).clip(.5,2)
vol=r.rolling(30,min_periods=20).std(); f=(-resid/(vol+1e-8)/scale.values[:,None]).shift(1)
for h in [5,10,20]:
 fr=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q); ns.append(len(z))
 x=pd.Series(vals); print('H',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for n in [365,730,1095]:
 fr=px.shift(-10)/px-1; vals=[]
 for dt in f.index[-n:]:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=pd.Series(vals); print('recent',n,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
print('coverage',round(f.notna().sum().sum()/px.notna().sum().sum(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'dates',len(px),'instruments',len(U))
