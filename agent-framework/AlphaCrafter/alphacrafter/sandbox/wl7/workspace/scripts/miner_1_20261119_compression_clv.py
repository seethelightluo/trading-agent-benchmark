import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy();d.date=pd.to_datetime(d.date).dt.normalize();return d.drop_duplicates('date').set_index('date').sort_index()
  except Exception:pass
D={s:get(s) for s in U};D={s:d for s,d in D.items() if d is not None}
def fac(d,h):
 c=d.close;o=d.open; r=c.pct_change(); v=r.rolling(20,min_periods=10).std(); vlong=r.rolling(80,min_periods=40).std()
 # buy assets with unusually compressed recent volatility, conditional on positive close location
 compression=(v/(vlong+1e-12)-1).clip(-3,3)
 clv=((c-d.low)/(d.high-d.low).replace(0,np.nan)*2-1)
 f=(-compression*0.7+clv.rolling(3,min_periods=2).mean()*0.3).shift(1)
 return pd.DataFrame({'f':f,'fr':c.shift(-h)/c-1}).dropna()
def calc(h,lo=None,hi=None):
 z=[]
 for s,d in D.items():q=fac(d,h).assign(asset=s).reset_index();z.append(q)
 q=pd.concat(z)
 if lo:q=q[q.date>=lo]
 if hi:q=q[q.date<=hi]
 a=[];ns=[]
 for _,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:a.append(g.f.corr(g.fr,method='spearman'));ns.append(len(g))
 a=pd.Series(a);return len(a),np.mean(ns),a.mean(),a.mean()/(a.std(ddof=1)+1e-12),np.mean(a>0)
print('assets',len(D),'span',min(d.index.min() for d in D.values()),max(d.index.max() for d in D.values()))
for h in [1,5,10,20]:print('horizon',h,calc(h))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-11-19')]:print('regime',lo,hi,calc(1,lo,hi))
