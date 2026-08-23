import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-01-27')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
 return None
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
C=pd.concat({s:d.close for s,d in D.items()},axis=1).sort_index(); R=C.pct_change()
breadth=R.gt(0).rolling(20,min_periods=12).mean().mean(axis=1)
def build(h):
 rows=[]
 for s,d in D.items():
  c=d.close; r=c.pct_change(); mom=c.pct_change(20); v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std()
  state=(v60/(v20+1e-8)).clip(0.5,1.5)
  f=(mom/(v20+1e-8)*(0.5+breadth)*state).shift(1); fr=c.shift(-h)/c-1
  rows.append(pd.DataFrame({'f':f,'fr':fr,'asset':s}))
 return pd.concat(rows).replace([np.inf,-np.inf],np.nan).dropna().reset_index(names='date')
def calc(q):
 a=[]; ns=[]
 for _,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: a.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 x=pd.Series(a); return len(x),np.mean(ns),x.mean(),x.mean()/x.std(ddof=1)*np.sqrt(252),(x>0).mean()
q=build(1); q[['date','asset','f']].to_csv('scripts/miner_3_20270128_vol_state_momentum_signal.csv',index=False)
print('assets',len(D),'range',min(d.index.min() for d in D.values()),max(d.index.max() for d in D.values()))
print('coverage',q.f.notna().mean(),'dates',q.date.nunique())
for h in [1,5,10,20]: print('horizon',h,calc(build(h)))
for lo,hi in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',lo,hi,calc(q[(q.date.dt.year>=lo)&(q.date.dt.year<=hi)]))
r=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',r.diff().abs().mean().mean())
