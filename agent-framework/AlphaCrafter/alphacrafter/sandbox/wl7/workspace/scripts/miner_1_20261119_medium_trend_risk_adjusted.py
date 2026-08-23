import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index()
  except Exception: pass
D={s:get(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
def factor(d,h):
 c=d.close; r=c.pct_change(); vol=r.rolling(40,min_periods=25).std()
 # medium-term trend adjusted for realized risk, lagged one day
 f=(c.pct_change(60)/(vol*np.sqrt(60)+1e-12)).shift(1)
 fr=c.shift(-h)/c-1
 return pd.DataFrame({'f':f,'fr':fr}).dropna()
def calc(h, start=None, end=None):
 z=[]
 for s,d in D.items():
  q=factor(d,h); q['asset']=s; z.append(q.reset_index())
 q=pd.concat(z)
 if start:q=q[q.date>=start]
 if end:q=q[q.date<=end]
 vals=[]; ns=[]
 for _,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: vals.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 a=pd.Series(vals); return len(a),np.mean(ns),a.mean(),a.mean()/(a.std(ddof=1)+1e-12),np.mean(a>0)
print('assets',len(D),'date span',min(d.index.min() for d in D.values()),max(d.index.max() for d in D.values()))
for h in [1,5,10,20]: print('horizon',h,'dates avg_names IC ICIR hit',calc(h))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-11-19')]: print('regime',lo,hi,calc(1,lo,hi))
print('coverage assets',len(D),'valid cross sections use >=8')
