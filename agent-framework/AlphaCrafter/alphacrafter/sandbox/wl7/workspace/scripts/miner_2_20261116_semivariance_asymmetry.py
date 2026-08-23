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
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
# Candidate: downside/upside semivariance asymmetry reversal. Positive values mean losses
# have dominated recent movement, so expect subsequent mean reversion.
def make(d,h):
 c=d.close.replace(0,np.nan); r=c.pct_change()
 dn=(-r.clip(upper=0)).pow(2).rolling(10,min_periods=7).mean()
 up=r.clip(lower=0).pow(2).rolling(10,min_periods=7).mean()
 f=((dn-up)/(dn+up+1e-10)).shift(1)
 fr=c.shift(-h)/c-1
 return pd.DataFrame({'f':f,'fr':fr}).replace([np.inf,-np.inf],np.nan).dropna()
def calc(h,lo=None,hi=None):
 rows=[]
 for s,d in D.items():
  q=make(d,h); q=q.loc[(q.index>=pd.Timestamp(lo or '2020-01-01'))&(q.index<=pd.Timestamp(hi or '2026-11-15'))]; q['asset']=s; rows.append(q.reset_index())
 q=pd.concat(rows); vals=[]; ns=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: vals.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 a=pd.Series(vals); return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(252),(a>0).mean()
print('assets',len(D),'date_span',min(d.index.min() for d in D.values()).date(),max(d.index.max() for d in D.values()).date())
for h in [1,5,10,20]: print('horizon',h,calc(h))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-11-15')]: print('regime',lo[:4]+'-'+hi[:4],calc(1,lo,hi))
rows=[make(d,1).f.rename(s) for s,d in D.items()]; x=pd.concat(rows,axis=1); ranks=x.rank(axis=1,pct=True)
print('turnover',ranks.diff().abs().mean(axis=1).mean(),'rank_dates',len(ranks),'coverage',len(ranks)/len(pd.date_range(ranks.index.min(),ranks.index.max(),freq='D')))
print('period',ranks.index.min().date(),ranks.index.max().date())
