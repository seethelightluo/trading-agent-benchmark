import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
arr=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END]
 arr.append(d.set_index('date').close.rename(s))
p=pd.concat(arr,axis=1); r=p.pct_change(); market=r.median(axis=1)
# rolling beta of each asset versus cross-sectional median, residual 5d return
f=[]
for s in syms:
 beta=r[s].rolling(60,min_periods=30).cov(market)/market.rolling(60,min_periods=30).var()
 r5=p[s].pct_change(5); m5=market.rolling(5).sum()
 resid=r5-beta*m5
 vol=r[s].rolling(20,min_periods=10).std()
 z=-resid/vol.replace(0,np.nan)
 f.append(z.rename(s))
factor=pd.concat(f,axis=1)
# forward returns
for h in [1,5,10]:
 y=p.shift(-h)/p-1; ics=[]; ns=[]
 for dt in factor.index:
  q=pd.concat([factor.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8: ics.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 a=np.asarray(ics); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
# regimes
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-12-17')]:
  ics=[]
  for dt in factor.loc[lo:hi].index:
   q=pd.concat([factor.loc[dt],(p.shift(-1)/p-1).loc[dt]],axis=1).dropna()
   if len(q)>=8: ics.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
  a=np.asarray(ics);print(lo,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6))
v=factor.stack().rename('factor').reset_index();v.columns=['date','symbol','factor'];v.to_csv('scripts/miner_3_20261217_betaneutral_reversal_signal.csv',index=False)
print('coverage',round(v.factor.notna().mean(),4),'rows',len(v))
# turnover
ranks=factor.rank(axis=1,pct=True);print('turnover',round(ranks.diff().abs().mean(axis=1).mean(),6))
