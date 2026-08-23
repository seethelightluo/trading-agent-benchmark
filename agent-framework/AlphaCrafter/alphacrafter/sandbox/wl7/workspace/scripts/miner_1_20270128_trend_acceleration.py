import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-01-27')
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,3000)
   if x is not None and len(x):
    x=x.copy(); x.date=pd.to_datetime(x.date).dt.normalize(); return x.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
D={s:get(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
R10=pd.concat({s:d.close.pct_change(10) for s,d in D.items()},axis=1); R30=pd.concat({s:d.close.pct_change(30) for s,d in D.items()},axis=1)
V=pd.concat({s:d.close.pct_change().rolling(30).std() for s,d in D.items()},axis=1)
# Recent trend acceleration relative to realized risk, lagged to avoid lookahead.
F=((R10-R30/3)/V).shift(1)
FR=pd.concat({s:D[s].close.shift(-1)/D[s].close-1 for s in D},axis=1)
def run(lo=None,hi=None):
 vals=[]; ns=[]
 for dt in F.index:
  if lo and dt<pd.Timestamp(lo): continue
  if hi and dt>pd.Timestamp(hi): continue
  q=pd.concat([F.loc[dt],FR.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1: vals.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman')); ns.append(len(q))
 a=pd.Series(vals); return len(a),float(np.mean(ns)),float(a.mean()),float(a.mean()/a.std(ddof=1)*np.sqrt(252)),float((a>0).mean())
print('assets',len(D),'dates',F.index.min().date(),F.index.max().date(),'overall',run(),'coverage',float(F.notna().mean().mean()))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-01-27'),('2026-07-16','2027-01-27')]: print(lo+' '+hi,run(lo,hi))
r=F.rank(axis=1,pct=True); print('turnover',float(r.diff().abs().mean(axis=1).mean())); r.to_csv('scripts/miner_1_20270128_trend_acceleration_signal.csv')
