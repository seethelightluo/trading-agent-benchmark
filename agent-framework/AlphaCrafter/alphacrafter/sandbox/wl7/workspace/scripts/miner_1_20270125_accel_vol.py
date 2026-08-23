import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUTOFF=pd.Timestamp('2027-01-22')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,3000)
   if x is not None and len(x):
    x=x.copy(); x.date=pd.to_datetime(x.date).dt.normalize(); return x.drop_duplicates('date').set_index('date').sort_index().loc[:CUTOFF]
  except Exception: pass
 return None
D={s:fetch(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
def fac(d,h):
 c=d.close.replace(0,np.nan); r=c.pct_change()
 fast=c/c.shift(5)-1; slow=c/c.shift(20)-1
 vol=r.rolling(20,min_periods=15).std()
 # Recent acceleration relative to the long trend, risk scaled; lagged one completed day
 f=((fast-0.25*slow)/(vol*np.sqrt(20)+1e-8)).shift(1)
 return pd.DataFrame({'f':f,'fr':c.shift(-h)/c-1}).replace([np.inf,-np.inf],np.nan).dropna()
def calc(h,lo=None,hi=None):
 z=[]
 for s,d in D.items():
  q=fac(d,h)
  if lo:q=q[(q.index>=pd.Timestamp(lo))&(q.index<=pd.Timestamp(hi))]
  q['asset']=s; z.append(q.reset_index())
 q=pd.concat(z); ics=[]; ns=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:
   ics.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 a=pd.Series(ics).dropna(); return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(252),(a>0).mean(),q.date.nunique()
print('assets',len(D),'lengths',min(map(len,D.values())),max(map(len,D.values())))
for h in [1,5,10,20]: print('horizon',h,calc(h))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-01-22')]: print('regime',lo[:4]+'-'+hi[:4],calc(1,lo,hi))
z=[]
for s,d in D.items(): z.append(fac(d,1).f.rename(s))
p=pd.concat(z,axis=1).rank(axis=1,pct=True); print('rank_dates',len(p),'coverage',p.notna().mean().mean(),'turnover',p.diff().abs().mean(axis=1).mean())
p.to_csv('scripts/miner_1_20270125_accel_vol_signal.csv')
