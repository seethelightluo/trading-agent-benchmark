import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index()
  except Exception: pass
D={s:fetch(s) for s in U}; D={s:d.loc[:'2026-11-14'] for s,d in D.items() if d is not None}
# Candidate: lagged 10d momentum, reward persistence and penalize downside volatility.
def make(d,h):
 c=d.close.replace(0,np.nan); r=c.pct_change()
 mom=c.pct_change(10)
 down=(-r.clip(upper=0)).rolling(20,min_periods=12).std()
 up=(r.clip(lower=0)).rolling(20,min_periods=12).std()
 pos=(r>0).rolling(10,min_periods=8).mean()
 f=(mom/(down+1e-8))*(0.5+0.5*pos)*(1/(1+up)).shift(1)
 fr=c.shift(-h)/c-1
 return pd.DataFrame({'f':f,'fr':fr}).replace([np.inf,-np.inf],np.nan).dropna()
def calc(h,lo='2020-01-01',hi='2026-11-14'):
 rows=[]
 for s,d in D.items():
  q=make(d,h).loc[lo:hi].copy(); q['asset']=s; rows.append(q.reset_index())
 q=pd.concat(rows); vals=[]; ns=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: vals.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 a=pd.Series(vals); return len(a),float(np.mean(ns)),float(a.mean()),float(a.mean()/a.std(ddof=1)*np.sqrt(252)),float((a>0).mean()),q.date.nunique()
print('assets',len(D),'minmax',min(len(x) for x in D.values()),max(len(x) for x in D.values()))
for h in [1,5,10,20]: print('horizon',h,calc(h))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-11-14')]: print('regime',lo[:4]+'-'+hi[:4],calc(1,lo,hi))
rows=[make(d,1).f.rename(s) for s,d in D.items()]; x=pd.concat(rows,axis=1); ranks=x.rank(axis=1,pct=True)
print('turnover',float(ranks.diff().abs().mean(axis=1).mean()),'rank_dates',len(ranks),'period',ranks.index.min().date(),ranks.index.max().date())
