import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-07-15'
F={}; C={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]
 ret=d.close.pct_change(); intr=(d.close/d.open-1).replace([np.inf,-np.inf],np.nan)
 F[s]=(intr.rolling(10,min_periods=7).mean()/ret.rolling(20,min_periods=10).std()).rename(s); C[s]=d.close
p=pd.DataFrame(C).sort_index(); fac=pd.DataFrame(F).reindex(p.index); fwd=p.shift(-1)/p-1

def evalh(y):
  ics=[];ns=[];dates=[]
  for dt in fac.index:
   z=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1:
    q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
    if np.isfinite(q):ics.append(q);ns.append(len(z));dates.append(pd.Timestamp(dt))
  a=np.asarray(ics); return a,np.asarray(ns),pd.DatetimeIndex(dates)
a,ns,dates=evalh(fwd); print('candidate intraday_pressure_10d');print('dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round(np.mean(a>0),4),'turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-15')]:
 q=a[(dates>=pd.Timestamp(lo))&(dates<=pd.Timestamp(hi))];print('regime',lo,'n',len(q),'IC',round(q.mean(),5),'ICIR',round(q.mean()/q.std(ddof=1),5))
for h in (5,10):
 aa,_,_=evalh(p.shift(-h)/p-1);print('horizon',h,'n',len(aa),'IC',round(aa.mean(),5),'ICIR',round(aa.mean()/aa.std(ddof=1),5))
for name,x in [('rev5',-(p/p.shift(5)-1)),('mom20',(p/p.shift(20)-1))]:print('corr',name,round(fac.stack().corr(x.stack()),5))
