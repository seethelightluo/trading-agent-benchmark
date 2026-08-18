import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
 d=get_stock_daily_data(s,2600)
 if d is None or len(d)<300: d=get_index_daily_data(s,2600)
 if d is not None and len(d): frames[s]=d.set_index('date')['close']
p=pd.concat(frames,axis=1).sort_index().ffill(); r=p.pct_change(); bench=r.mean(axis=1)
bvar=bench.rolling(40,min_periods=30).var(); beta=pd.DataFrame(index=r.index,columns=r.columns,dtype=float)
for s in r.columns: beta[s]=r[s].rolling(40,min_periods=30).cov(bench)/bvar
res=r-beta.mul(bench,axis=0); res5=res.rolling(5,min_periods=5).sum(); disp=r.rolling(20,min_periods=15).std().mean(axis=1)
intensity=(disp/disp.rolling(60,min_periods=40).median()).clip(.5,2); f=(-res5.mul(intensity,axis=0)).shift(1)
print('rows',len(p),'assets',len(frames),'range',p.index.min(),p.index.max(),'coverage',round(f.notna().mean().mean(),4))
for h in [1,5,10,20]:
 fr=p.pct_change(h).shift(-h); vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 a=np.array(vals); a=a[np.isfinite(a)]
 print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round(np.mean(a>0),4))
fr=p.pct_change(10).shift(-10); vals=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
a=pd.Series(dict(vals))
for n in [250,500]:
 q=a.tail(n); print('recent',n,'dates',len(q),'IC',round(q.mean(),5),'ICIR',round(q.mean()/q.std(ddof=1),5))
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),5))
