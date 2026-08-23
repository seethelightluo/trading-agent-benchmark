import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2026-12-02')
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,3000)
   if x is not None and len(x):
    x=x.copy(); x.date=pd.to_datetime(x.date).dt.normalize(); return x.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
D={s:get(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
# Novel interpretable factor: lagged 10-day trend, benchmark-relative and volatility scaled.
# Relative trend is intended to remove common market beta; lag avoids lookahead.
R=pd.concat({s:d.close.pct_change(10) for s,d in D.items()},axis=1)
V=pd.concat({s:d.close.pct_change().rolling(20).std() for s,d in D.items()},axis=1)
F=R.sub(R.median(axis=1),axis=0).div(V.replace(0,np.nan)).shift(1)
def evaluate(h, start=None,end=None):
 Y=pd.concat({s:d.close.shift(-h)/d.close-1 for s,d in D.items()},axis=1)
 vals=[]; ns=[]
 for dt in F.index:
  if start and dt<pd.Timestamp(start): continue
  if end and dt>pd.Timestamp(end): continue
  q=pd.concat([F.loc[dt],Y.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   vals.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman')); ns.append(len(q))
 a=pd.Series(vals)
 return len(a),float(np.mean(ns)),float(a.mean()),float(a.mean()/a.std(ddof=1)*np.sqrt(252)),float((a>0).mean())
print('assets',len(D),'dates',len(F),'cutoff',CUT.date())
for h in [1,5,10,20]: print('horizon',h,evaluate(h))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-12-02')]: print('regime',lo[:4]+'-'+hi[:4],evaluate(1,lo,hi))
print('coverage',float(F.notna().mean().mean()),'turnover',float(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
F.to_csv('scripts/miner_3_20261202_relative_trend_signal.csv')
