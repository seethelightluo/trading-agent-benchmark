"""Validate one factor: asset-local short/long realized-volatility ratio."""
import os,numpy as np,pandas as pd
DATA='../persistent/stock_data'; END=pd.Timestamp('2026-07-15')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
prices={}
for s in A:
 d=pd.read_csv(os.path.join(DATA,s+'.csv'),parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 prices[s]=d.close.astype(float).where(d.close>0).loc[:END]
r={s:p.pct_change() for s,p in prices.items()}
sig=pd.concat({s:(x.rolling(5,min_periods=5).std()/x.rolling(60,min_periods=40).std()) for s,x in r.items()},axis=1)
for h in [1,5,10,20]:
 fw=pd.concat({s:p.pct_change(h).shift(-h) for s,p in prices.items()},axis=1)
 vals=[];ns=[];dates=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));dates.append(dt)
 x=pd.Series(vals,index=dates);print('horizon',h,'dates',len(x),'mean_valid',np.mean(ns),'coverage',np.mean(ns)/15,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
fw=pd.concat({s:p.pct_change(10).shift(-10) for s,p in prices.items()},axis=1);vals=[];dates=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
 if len(z)>=8:vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));dates.append(dt)
x=pd.Series(vals,index=dates)
for n,a,b in [('2020','2020','2020'),('2021_22','2021','2022'),('2023_24','2023','2024'),('2025_26','2025','2026')]:
 q=x.loc[a:b];print('regime',n,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
turn=[]
for a,b in zip(sig.index[:-1],sig.index[1:]):
 z=pd.concat([sig.loc[a],sig.loc[b]],axis=1).dropna()
 if len(z)>=8:turn.append(np.abs(z.iloc[:,0].rank(pct=True)-z.iloc[:,1].rank(pct=True)).mean())
print('turnover',np.mean(turn),'turnover_dates',len(turn))
for n, f in [('ravmom',pd.concat({s:(p/p.shift(20)-1)/r[s].rolling(20,min_periods=15).std() for s,p in prices.items()},axis=1)),('reversal',pd.concat({s:-(p/p.shift(5)-1)/r[s].rolling(5,min_periods=4).std() for s,p in prices.items()},axis=1)),('vol',pd.concat({s:r[s].rolling(20,min_periods=15).std() for s in prices},axis=1))]:
 z=pd.concat([sig.stack(),f.stack()],axis=1).dropna();print('corr',n,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),'cells',len(z))
