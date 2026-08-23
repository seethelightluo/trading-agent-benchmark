import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];CUT=pd.Timestamp('2027-01-27')
def g(s):
 for f in(get_index_daily_data,get_stock_daily_data):
  try:
   x=f(s,3000)
   if x is not None and len(x):
    x=x.copy();x.date=pd.to_datetime(x.date).dt.normalize();return x.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except:pass
D={s:g(s) for s in U};D={s:x for s,x in D.items() if x is not None}
R20=pd.concat({s:x.close.pct_change(20) for s,x in D.items()},axis=1); R60=pd.concat({s:x.close.pct_change(60) for s,x in D.items()},axis=1); V=pd.concat({s:x.close.pct_change().rolling(40).std() for s,x in D.items()},axis=1); C=pd.concat({s:x.close for s,x in D.items()},axis=1)
F=((R20-R60/3)/V).shift(1); FR={h:pd.concat({s:x.close.shift(-h)/x.close-1 for s,x in D.items()},axis=1) for h in [1,5,10,20]}
def run(X,h=1,lo=None,hi=None):
 a=[];ns=[]
 for dt in X.index:
  if lo and dt<pd.Timestamp(lo):continue
  if hi and dt>pd.Timestamp(hi):continue
  q=pd.concat([X.loc[dt],FR[h].loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:a.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'));ns.append(len(q))
 z=pd.Series(a);return len(z),np.mean(ns),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252),(z>0).mean()
print('assets',len(D),'dates',F.index.min().date(),F.index.max().date(),'overall',run(F),'coverage',F.notna().mean().mean())
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-01-27')]:print(lo[:4]+'-'+hi[:4],run(F,1,lo,hi))
print('decay',{h:run(F,h) for h in [1,5,10,20]});print('turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean());F.rank(axis=1,pct=True).to_csv('scripts/miner_3_20270128_trend_acceleration_signal.csv')
