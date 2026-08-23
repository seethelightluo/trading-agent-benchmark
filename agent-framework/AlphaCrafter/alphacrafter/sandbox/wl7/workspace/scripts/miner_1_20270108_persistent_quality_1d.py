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
# Persistent-trend quality: lagged 20d return / realized vol, weighted by positive-day share.
F={}; Y={}
for s,d in D.items():
 c=d.close; r=c.pct_change(); vol=r.rolling(20).std(); pos=(r>0).rolling(20).mean()
 F[s]=(c.pct_change(20)/vol*np.sqrt(20)*(0.5+pos)).shift(1)
 Y[s]=c.shift(-1)/c-1
F=pd.concat(F,axis=1);Y=pd.concat(Y,axis=1)
def ev(lo=None,hi=None):
 a=[];ns=[]
 for dt in F.index:
  if lo and dt<pd.Timestamp(lo):continue
  if hi and dt>pd.Timestamp(hi):continue
  q=pd.concat([F.loc[dt],Y.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:a.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'));ns.append(len(q))
 a=pd.Series(a); return len(a),round(float(np.mean(ns)),2),round(float(a.mean()),6),round(float(a.mean()/a.std(ddof=1)*np.sqrt(252)),4),round(float((a>0).mean()),4)
print('assets',len(D),'factor_dates',len(F),'range',F.index.min().date(),F.index.max().date())
print('horizon10',ev())
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-01-07')]:print('regime',lo[:4]+'-'+hi[:4],ev(lo,hi))
print('coverage',round(float(F.notna().mean().mean()),4),'turnover',round(float(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()),4))
F.to_csv('scripts/miner_1_20270108_persistent_quality_signal.csv')
