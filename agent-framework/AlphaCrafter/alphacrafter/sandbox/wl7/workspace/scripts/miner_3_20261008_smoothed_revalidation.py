import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2026-10-07')
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d[d.date<=CUT].drop_duplicates('date').set_index('date').sort_index()
  except: pass
D={s:get(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
def factor(d,h=1):
 o=d.open.replace(0,np.nan); c=d.close; hl=(d.high-d.low)/o; med=hl.rolling(20,min_periods=10).median()
 f=(-(c/o-1)*(hl/(med+1e-8)).clip(0,4)).ewm(span=3,min_periods=3,adjust=False).mean().shift(1)
 return pd.DataFrame({'f':f,'fr':c.shift(-h)/c-1}).dropna()
def run(h,lo=None,hi=None):
 z=[]
 for s,d in D.items():
  q=factor(d,h).reset_index(); q['asset']=s; z.append(q)
 q=pd.concat(z); 
 if lo is not None:q=q[(q.date>=lo)&(q.date<=hi)]
 a=[]; ns=[]
 for _,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:a.append(g.f.corr(g.fr,method='spearman'));ns.append(len(g))
 a=pd.Series(a); return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(252),(a>0).mean()
print('assets',len(D),'cutoff',CUT,'range',min(x.index.min() for x in D.values()),max(x.index.max() for x in D.values()))
for h in [1,5,10,20]:print('h',h,run(h))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-10-07')]:print('regime',lo,hi,run(1,pd.Timestamp(lo),pd.Timestamp(hi)))
