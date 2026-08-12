import pandas as pd,numpy as np,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2030-12-12');D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv');x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close
p=pd.DataFrame(D).sort_index().loc[:end];r=p.pct_change(); fr=p.shift(-1)/p-1
for n in [5,10,20,40,60]:
 f=(-r.rolling(n).std()).shift(1);ics=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8:ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 x=pd.Series(ics).dropna();print(n,len(x),np.mean(ns),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
