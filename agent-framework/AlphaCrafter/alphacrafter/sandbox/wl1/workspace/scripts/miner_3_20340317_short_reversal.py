import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0:d=get_index_daily_data(s,5000)
 if d is not None and len(d):P[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
px=pd.DataFrame(P).sort_index().ffill();r=px.pct_change();v=r.rolling(20,min_periods=15).std()*np.sqrt(20)
# Candidate: short-horizon reversal, damped when recent path is noisy.
ret5=np.log(px/px.shift(5)); path=(r.rolling(20,min_periods=15).sum().abs()/(r.abs().rolling(20,min_periods=15).sum()+1e-12)).clip(0,1)
f=(-ret5/(v+1e-6))*(0.5+path);f=f.shift(1);fr=px.pct_change(10).shift(-10)
ics=[];ns=[];ds=[]
for dt in f.index:
 a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(a)>=8:
  c=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
  if np.isfinite(c):ics.append(c);ns.append(len(a));ds.append(dt)
z=np.array(ics);rank=f.rank(axis=1,pct=True)
print('dates',len(z),'avgN %.2f'%np.mean(ns),'IC %.8f ICIR %.8f hit %.5f coverage %.5f turnover %.5f'%(z.mean(),z.mean()/z.std(ddof=1),np.mean(z>0),f.notna().sum(axis=1).mean()/len(U),rank.diff().abs().mean(axis=1).dropna().mean()))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 q=np.array([v for d,v in zip(ds,ics) if a<=str(d.year)<=b]);print('REG',a,b,len(q),'IC %.8f ICIR %.8f'%(q.mean(),q.mean()/q.std(ddof=1)))
f.loc[ds].to_csv('scripts/miner_3_20340317_short_reversal_signal.csv',index_label='date')
