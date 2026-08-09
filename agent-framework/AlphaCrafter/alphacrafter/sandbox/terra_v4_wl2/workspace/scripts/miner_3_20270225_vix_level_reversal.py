import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def read(s):
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x['date']=pd.to_datetime(x['date']); return x.set_index('date')['close']
px=pd.concat({s:read(s) for s in U},axis=1).sort_index(); r=px.pct_change(); v=pd.read_csv('../persistent/index_data/VIX.csv');v['date']=pd.to_datetime(v.date); vv=v.set_index('date').close.pct_change().reindex(px.index).ffill(); vl=v.set_index('date').close.reindex(px.index).ffill()
# lagged high-volatility level, 3d reversal; regime threshold is rolling 80th percentile, all lagged
high=(vl>vl.rolling(252,min_periods=100).quantile(.80)).shift(1)
f=(-r.rolling(3).sum()).where(high); f=f.sub(f.median(axis=1),axis=0)
fr=px.shift(-1)/px-1
A=[]; ns=[]; dates=[]
for d in f.index:
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:A.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));dates.append(d)
a=np.array(A); print('dates',len(a),'avgN',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0),'coverage',len(a)/len(px))
for h in [5,10]:
 q=[]
 for d in f.index:
  z=pd.concat([f.loc[d],(px.shift(-h)/px-1).loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=np.array(q);print('h',h,'dates',len(q),'IC',np.mean(q),'ICIR',np.mean(q)/np.std(q,ddof=1))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2027-02-25')]:
 q=[]
 for d in f.loc[lo:hi].index:
  z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=np.array(q);print(lo, len(q), np.mean(q) if len(q) else np.nan, np.mean(q)/np.std(q,ddof=1) if len(q)>1 else np.nan)
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_3_20270225_vix_level_reversal3.csv',index=False)
