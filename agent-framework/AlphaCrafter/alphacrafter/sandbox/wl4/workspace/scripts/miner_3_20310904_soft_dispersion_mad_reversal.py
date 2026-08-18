import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2031-09-03'); p={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()['close']; p[s]=d[d.index<=cutoff]
px=pd.DataFrame(p).sort_index(); r=px.pct_change(); L=15; W=40
res=r.rolling(L,min_periods=10).sum(); res=res.sub(res.median(axis=1),axis=0)
med=r.rolling(W,min_periods=20).median(); mad=(r-med).abs().rolling(W,min_periods=20).median()*1.4826
vol=r.rolling(W,min_periods=20).std(); scale=.7*mad+.3*vol
base=-res/(scale+1e-8)
disp=r.rolling(20,min_periods=10).std().mean(axis=1)
ref=disp.rolling(252,min_periods=60).median()
# Soft regime weight: low dispersion retains half signal, high dispersion reaches 1.5x; lag all inputs.
weight=(disp/(ref+1e-12)).clip(.5,1.5)
f=base.mul(weight,axis=0).shift(1)
fr={h:px.shift(-h)/px-1 for h in [5,10,20]}
for h in [5,10,20]:
 vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr[h].loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 x=pd.Series(vals); print('H',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for n in [365,730,1095]:
 vals=[]
 for dt in f.index[-n:]:
  z=pd.concat([f.loc[dt],fr[10].loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=pd.Series(vals); print('recent',n,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'dates',len(x))
print('coverage',round(f.notna().sum().sum()/px.notna().sum().sum(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'instruments',len(U),'dates',len(px))
