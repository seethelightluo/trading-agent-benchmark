import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut='2026-07-15'; P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); P[s]=d.close.loc[:cut]
p=pd.DataFrame(P).sort_index(); r=p.pct_change(); vx=r['SPX']; vv=vx.rolling(60,min_periods=20).var() 
b=pd.DataFrame(index=p.index,columns=U,dtype=float)
for s in U: b[s]=r[s].rolling(60,min_periods=20).cov(vx)/vv
res=r-b.mul(vx,axis=0); print('r',r.shape,'b',b.shape,'res',res.shape,'bcols',list(b.columns)[:3]); fac=res.rolling(20,min_periods=10).sum(); fwd=p.shift(-1)/p-1
ics=[]; ns=[]; dates=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q): ics.append(q);ns.append(len(z));dates.append(dt)
a=np.asarray(ics)
print('dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round(np.mean(a>0),4),'turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-15')]:
 q=a[(pd.DatetimeIndex(dates)>=pd.Timestamp(lo))&(pd.DatetimeIndex(dates)<=pd.Timestamp(hi))]; print('regime',lo,'n',len(q),'IC',round(q.mean(),5),'ICIR',round(q.mean()/q.std(ddof=1),5))
for h in (5,10):
 y=p.shift(-h)/p-1; aa=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 aa=np.asarray(aa);print('horizon',h,'n',len(aa),'IC',round(aa.mean(),5),'ICIR',round(aa.mean()/aa.std(ddof=1),5))
for name,x in [('rev5',-(p/p.shift(5)-1)),('rev3',-(p/p.shift(3)-1)),('mom20',(p/p.shift(20)-1))]: print('corr',name,round(fac.stack().corr(x.stack()),5))
