import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-02-21')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,3000)
   if x is not None and len(x):
    x=x.copy(); x.date=pd.to_datetime(x.date).dt.normalize(); return x.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
 return None
D={s:fetch(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
C=pd.concat({s:x.close for s,x in D.items()},axis=1).sort_index(); R=C.pct_change(); breadth=R.gt(0).rolling(20,min_periods=10).mean().mean(axis=1)
def make(h):
 rows=[]
 for s,d in D.items():
  c=d.close; r=c.pct_change(); vol=r.rolling(20,min_periods=12).std()
  # reversal is strongest when market breadth is near neutral; lag all inputs
  neutral=(1-(breadth-0.5).abs()*2).clip(0,1)
  f=((-c.pct_change(5)/(vol*np.sqrt(20)+1e-8))*(0.5+neutral)).shift(1)
  fr=c.shift(-h)/c-1
  rows.append(pd.DataFrame({'f':f,'fr':fr,'asset':s}))
 return pd.concat(rows).replace([np.inf,-np.inf],np.nan).dropna().reset_index(names='date')
def calc(q):
 vals=[]; ns=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: vals.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 x=pd.Series(vals); return len(x),np.mean(ns),x.mean(),x.mean()/x.std(ddof=1)*np.sqrt(252),(x>0).mean()
print('assets',len(D),'date_end',max(x.index.max() for x in D.values()))
for h in [1,5,10,20]: print('horizon',h,calc(make(h)))
q=make(1)
for lo,hi in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',lo,hi,calc(q[(q.date.dt.year>=lo)&(q.date.dt.year<=hi)]))
print('coverage',q.f.notna().mean(),'dates',q.date.nunique(),'avg assets',q.groupby('date').size().mean())
# rank turnover
z=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',z.diff().abs().mean().mean())
