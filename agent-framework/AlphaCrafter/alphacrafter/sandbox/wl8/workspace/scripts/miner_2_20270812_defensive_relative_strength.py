import os, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
prices={}
for a in ASSETS:
    d=pd.read_csv(os.path.join(base,a+'.csv'))
    d['date']=pd.to_datetime(d['date'])
    d=d.sort_values('date').drop_duplicates('date').set_index('date')
    prices[a]=d['close'].astype(float)
p=pd.concat(prices,axis=1).sort_index()
# Candidate: lagged 10-observation relative strength, with 3-observation smoothing.
r=p.pct_change(10)
csmed=r.median(axis=1)
f=r.sub(csmed,axis=0).rolling(3,min_periods=3).mean().shift(1)
# forward one-day returns, only information available at t-1 represented by f at t
fr=p.pct_change(1).shift(-1)
rows=[]
for dt in f.index:
    x=f.loc[dt]; y=fr.loc[dt]
    z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
        ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
        rows.append((dt,ic,len(z)))
rw=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
mean=rw.ic.mean(); sd=rw.ic.std(ddof=1); icir=mean/sd*np.sqrt(252) if sd else np.nan
print('candidate=10d_relative_strength_3d_smooth')
print('dates',len(rw),'rows',int(rw.n.sum()),'avg_n',rw.n.mean(),'coverage',rw.n.sum()/(len(rw)*15))
print('IC',mean,'ICIR',icir,'hit',float((rw.ic>0).mean()))
for name,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-25','2023-01-01','2025-12-31'),('2026','2026-01-01','2026-12-31'),('2027YTD','2027-01-01','2027-08-11'),('recent90','2027-05-01','2027-08-11')]:
 q=rw.loc[lo:hi,'ic']; print(name,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252) if len(q)>1 and q.std(ddof=1)>0 else np.nan)
# turnover: rank top/bottom signal changes daily
rank=f.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna()
print('turnover_proxy',turn.mean(),'decay_5d_pending')
for h in [5,10]:
 fy=p.pct_change(h).shift(-1) # approximate forward h from t to t+h
 vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fy.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('horizon',h,'dates',len(vals),'IC',np.mean(vals),'ICIR',np.mean(vals)/np.std(vals,ddof=1)*np.sqrt(252) if len(vals)>1 else np.nan)
