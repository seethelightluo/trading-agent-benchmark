import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2026-12-11')
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,3000)
   if x is not None and len(x):
    x=x.copy();x.date=pd.to_datetime(x.date).dt.normalize();return x.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except: pass
D={s:get(s) for s in U};D={s:x for s,x in D.items() if x is not None}
# Relative momentum: lagged 20d return minus cross-sectional median, normalized by idiosyncratic volatility.
R=pd.concat({s:d.close.pct_change(20) for s,d in D.items()},axis=1); V=pd.concat({s:d.close.pct_change().rolling(20).std() for s,d in D.items()},axis=1)
med=R.median(axis=1); F=(R.sub(med,axis=0)/V).shift(1); FR=pd.concat({s:D[s].close.shift(-1)/D[s].close-1 for s in D},axis=1)
def run(h,lo=None,hi=None):
 fr=pd.concat({s:D[s].close.shift(-h)/D[s].close-1 for s in D},axis=1); z=F; vals=[];ns=[]
 for dt in z.index:
  if lo and dt<pd.Timestamp(lo):continue
  if hi and dt>pd.Timestamp(hi):continue
  q=pd.concat([z.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1: vals.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'));ns.append(len(q))
 a=pd.Series(vals);return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(252),(a>0).mean()
print('assets',len(D),'dates',len(F),'min_history',min(len(x) for x in D.values()))
for h in [1,5,10,20]:print('horizon',h,run(h))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-12-11')]:print('regime',lo[:4]+'-'+hi[:4],run(1,lo,hi))
print('coverage',F.notna().mean().mean(),'turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
F.to_csv('scripts/miner_3_20261214_relative_momentum_signal.csv')
