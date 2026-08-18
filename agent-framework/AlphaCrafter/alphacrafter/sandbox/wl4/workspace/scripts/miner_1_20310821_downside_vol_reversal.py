import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2031-08-20')
p={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close;p[s]=d[d.index<=cutoff]
p=pd.DataFrame(p).sort_index();r=p.pct_change();L,V=10,40
res=r.rolling(L).sum();res=res.sub(res.median(axis=1),axis=0)
down=r.where(r<0,0).rolling(V,min_periods=20).std(); vol=.7*down+.3*r.rolling(V,min_periods=20).std()
f=(-res/(vol+1e-8)).shift(1)
for h in [5,10,20]:
 fr=p.shift(-h)/p-1; vals=[];ns=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(q)>=8: vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 x=pd.Series(vals);print('H',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for n in [365,730,1095]:
 fr=p.shift(-10)/p-1;vals=[]
 for dt in f.index[-n:]:
  q=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(q)>=8:vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 x=pd.Series(vals);print('recent',n,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'dates_used',len(x))
print('coverage',round(f.notna().sum().sum()/p.notna().sum().sum(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4),'instruments',len(U),'dates',len(p))
