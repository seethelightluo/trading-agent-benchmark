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
C=pd.concat({s:d.close for s,d in D.items()},axis=1); ret=C.pct_change()
# Bull market: follow 20d relative momentum. Bear market: use short-term reversal.
trend=C['SPX'].pct_change(20)
vol=ret.rolling(20).std()*np.sqrt(20)
rel=C.pct_change(10).sub(C.pct_change(10).median(axis=1),axis=0)
mom=rel/vol
rev=-C.pct_change(3)/ret.rolling(15).std()
F=mom.where(trend>=0,rev).shift(1)
FR=C.shift(-1)/C-1

def run(lo=None,hi=None,h=1):
 a=[]; ns=[]
 f=F; fr=C.shift(-h)/C-1
 for dt in f.index:
  if lo and dt<pd.Timestamp(lo): continue
  if hi and dt>pd.Timestamp(hi): continue
  q=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1: a.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman')); ns.append(len(q))
 a=pd.Series(a); return len(a),float(np.mean(ns)),float(a.mean()),float(a.mean()/a.std(ddof=1)*np.sqrt(252)),float((a>0).mean())
print('assets',len(D),'dates',F.index.min().date(),F.index.max().date(),'overall',run(),'coverage',float(F.notna().mean().mean()))
for h in [2,5,10,20]: print('horizon',h,run(h=h))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-01-27'),('2026-07-16','2027-01-27')]: print(lo+' '+hi,run(lo,hi))
print('turnover',float(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
F.rank(axis=1,pct=True).to_csv('scripts/miner_1_20270128_regime_switch_signal.csv')
