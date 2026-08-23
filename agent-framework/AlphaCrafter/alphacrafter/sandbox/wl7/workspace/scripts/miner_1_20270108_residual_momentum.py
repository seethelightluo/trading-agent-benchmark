import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-01-07')
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,3000)
   if x is not None and len(x):
    x=x.copy(); x.date=pd.to_datetime(x.date).dt.normalize(); return x.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
D={s:get(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
C=pd.concat({s:d.close for s,d in D.items()},axis=1).sort_index()
# Residual 5-day momentum: asset return less contemporaneous cross-asset median, volatility scaled; lagged.
R=C.pct_change(5); V=C.pct_change().rolling(20).std(); breadth=(R>0).mean(axis=1)
F=R.sub(R.median(axis=1),axis=0).div(V.replace(0,np.nan)).mul(1+0.25*(breadth-0.5),axis=0).shift(1)
def ev(h,lo=None,hi=None):
 Y=C.shift(-h).div(C)-1; a=[]; ns=[]
 for dt in F.index:
  if lo and dt<pd.Timestamp(lo): continue
  if hi and dt>pd.Timestamp(hi): continue
  q=pd.concat([F.loc[dt],Y.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: a.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman')); ns.append(len(q))
 a=pd.Series(a)
 return len(a),round(float(np.mean(ns)),2),round(float(a.mean()),6),round(float(a.mean()/a.std(ddof=1)*np.sqrt(252)),4),round(float((a>0).mean()),4)
print('assets',len(D),'dates',len(F),'range',F.index.min().date(),F.index.max().date())
for h in [1,5,10,20]: print('horizon',h,ev(h))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-01-07')]: print('regime',lo[:4]+'-'+hi[:4],ev(1,lo,hi))
print('coverage',round(float(F.notna().mean().mean()),4),'turnover',round(float(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()),4))
F.to_csv('scripts/miner_1_20270108_residual_momentum_signal.csv')
