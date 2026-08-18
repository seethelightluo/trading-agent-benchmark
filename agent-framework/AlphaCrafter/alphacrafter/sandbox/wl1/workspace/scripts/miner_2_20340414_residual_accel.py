import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0:d=get_index_daily_data(s,5000)
 if d is not None and len(d):P[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
px=pd.DataFrame(P).sort_index().ffill(); r=np.log(px).diff()
# Cross-sectional residual acceleration: recent 5d reversal relative to 30d trend, demeaned each date.
raw=-(r.rolling(5,min_periods=5).sum()) + r.rolling(30,min_periods=20).sum()/6
f=raw.sub(raw.mean(axis=1),axis=0).shift(1); y=px.pct_change(10).shift(-10)
ics=[];ds=[];ns=[]
for dt in f.index:
 a=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(a)>=8:
  c=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
  if np.isfinite(c):ics.append(c);ds.append(dt);ns.append(len(a))
z=np.asarray(ics); rank=f.rank(axis=1,pct=True)
print('dates',len(z),'avgN',np.mean(ns),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0),'coverage',f.notna().sum(axis=1).mean()/len(U),'turnover',rank.diff().abs().mean(axis=1).dropna().mean())
for lo,hi in [(2024,2026),(2027,2029),(2030,2032),(2033,2034)]:
 q=np.array([v for d,v in zip(ds,ics) if lo<=d.year<=hi]);print('REG',lo,hi,len(q),q.mean(),q.mean()/q.std(ddof=1))
f.loc[ds].to_csv('scripts/miner_2_20340414_residual_accel_signal.csv',index_label='date')
