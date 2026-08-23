import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2026-12-02')
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,3000)
   if x is not None and len(x):
    x=x.copy();x.date=pd.to_datetime(x.date).dt.normalize();return x.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except:pass
D={s:get(s) for s in U};D={s:x for s,x in D.items() if x is not None}
# Volume-shock reversal: lagged one-day reversal weighted by abnormal log-volume, with volatility normalization.
ret=pd.concat({s:d.close.pct_change() for s,d in D.items()},axis=1)
vol=pd.concat({s:np.log1p(d.volume.replace(0,np.nan)) for s,d in D.items()},axis=1)
vs=vol.sub(vol.rolling(40,min_periods=20).median()).div(vol.rolling(40,min_periods=20).std())
rv=ret.rolling(20).std()
F=(-ret*vs.clip(lower=0)).div(rv).shift(1)
def ev(h,lo=None,hi=None):
 Y=pd.concat({s:d.close.shift(-h)/d.close-1 for s,d in D.items()},axis=1); a=[];ns=[]
 for dt in F.index:
  if lo and dt<pd.Timestamp(lo):continue
  if hi and dt>pd.Timestamp(hi):continue
  q=pd.concat([F.loc[dt],Y.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:a.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'));ns.append(len(q))
 a=pd.Series(a);return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(252),(a>0).mean()
print('assets',len(D),'cutoff',CUT.date())
for h in [1,5,10,20]:print('horizon',h,ev(h))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-12-02')]:print('regime',lo[:4]+'-'+hi[:4],ev(1,lo,hi))
print('coverage',F.notna().mean().mean(),'turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
