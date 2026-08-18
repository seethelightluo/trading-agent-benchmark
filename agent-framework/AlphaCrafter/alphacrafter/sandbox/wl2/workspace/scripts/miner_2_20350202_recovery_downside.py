import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; o={}
for s in U:
 d=get_stock_daily_data(s,days=6000)
 if d is None or len(d)<300:d=get_index_daily_data(s,days=6000)
 o[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(o).sort_index().ffill(); r=p.pct_change(); low=p.shift(1).rolling(60,min_periods=40).min(); rec=p.shift(1)/low-1; neg=r.shift(1).where(r.shift(1)<0).rolling(30,min_periods=20).std(); fac=(rec/(neg*np.sqrt(252)+1e-8)).replace([np.inf,-np.inf],np.nan)
print('assets',len(p.columns),'dates',len(p),'coverage',fac.notna().sum().sum()/(len(fac)*len(U)))
for h in [5,10,20,40]:
 fwd=p.shift(-h)/p-1; vals=[];ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=pd.Series(vals).dropna(); print('h dates avgN IC ICIR hit',h,len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(len(a)),(a>0).mean())
fwd=p.shift(-20)/p-1; vals=[];dates=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));dates.append(dt)
a=pd.Series(vals,index=pd.to_datetime(dates))
for lo,hi in [('2026','2030'),('2031','2035'),('2034','2035')]:
 q=a[(a.index>=pd.Timestamp(lo))&(a.index<=pd.Timestamp(hi+'-12-31'))];print('regime',lo,hi,len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean())
fac.to_csv('../persistent/miner_2_20350202_recovery_downside_signal.csv')
