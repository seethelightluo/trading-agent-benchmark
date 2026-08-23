import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index()
  except: pass
D={s:get(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
def series(d,h=1):
 o=d.open.replace(0,np.nan); c=d.close; hl=(d.high-d.low)/o
 med=hl.rolling(20,min_periods=10).median(); raw=-(c/o-1)*(hl/(med+1e-8)).clip(0,4)
 # smoother reduces daily rank noise while retaining lagged information
 f=raw.ewm(span=3,min_periods=3,adjust=False).mean().shift(1)
 fr=c.shift(-h)/c-1
 return pd.DataFrame({'f':f,'fr':fr}).dropna()
def calc(h):
 z=[]
 for s,d in D.items():
  q=series(d,h); q['asset']=s; z.append(q.reset_index())
 q=pd.concat(z); vals=[]; ns=[]
 for _,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: vals.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 a=pd.Series(vals); return len(a),float(np.mean(ns)),float(a.mean()),float(a.mean()/a.std(ddof=1)*np.sqrt(252)),float((a>0).mean())
print('assets',len(D),'dates',min(len(d) for d in D.values()),max(len(d) for d in D.values()))
for h in [1,5,10,20]: print('horizon',h,'dates avg_names IC ICIR hit',calc(h))
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 z=[]
 for s,d in D.items():
  q=series(d,1); q=q[(q.index.year>=lo)&(q.index.year<=hi)]; z.append(q.reset_index())
 q=pd.concat(z); a=[]
 for _,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:a.append(g.f.corr(g.fr,method='spearman'))
 a=pd.Series(a); print('regime',lo,hi,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1)*np.sqrt(252))
print('coverage',q.date.nunique() if len(q) else 0)
