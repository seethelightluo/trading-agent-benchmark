import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-03-21')
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy();d.date=pd.to_datetime(d.date).dt.normalize();return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
D={s:get(s) for s in U};D={s:d for s,d in D.items() if d is not None}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()})
# Equal-weight rank agreement over 10/30/60d, lagged to avoid lookahead.
raw=pd.concat({h:P.pct_change(h) for h in [10,30,60]},axis=1)
F=sum(raw[h].rank(axis=1,pct=True) for h in [10,30,60])/3
F=F.shift(1); FR={h:P.shift(-h)/P-1 for h in [1,5,10,20]}
def stats(f,h,lo=None,hi=None):
 z=[];ns=[]
 for dt in f.index:
  if lo and dt<pd.Timestamp(lo):continue
  if hi and dt>pd.Timestamp(hi):continue
  q=pd.concat([f.loc[dt],FR[h].loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:z.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'));ns.append(len(q))
 z=pd.Series(z);return len(z),float(np.mean(ns)),float(z.mean()),float(z.mean()/z.std(ddof=1)*np.sqrt(252)),float((z>0).mean())
print('assets',len(D),'dates',F.index.min().date(),F.index.max().date(),'coverage',float(F.notna().mean().mean()))
for h in [1,5,10,20]:print('horizon',h,stats(F,h))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-03-21'),('2026-07-16','2027-03-21')]:print('regime',lo,hi,stats(F,1,lo,hi))
print('turnover',float(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
F.to_csv('scripts/miner_1_20270322_multihorizon_agreement_signal.csv')
