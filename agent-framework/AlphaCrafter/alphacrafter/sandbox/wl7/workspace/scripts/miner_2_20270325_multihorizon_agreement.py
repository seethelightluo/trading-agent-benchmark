import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-03-24')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,3000)
   if x is not None and len(x):
    x=x.copy(); x['date']=pd.to_datetime(x['date']).dt.normalize(); return x.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
 return None
D={s:fetch(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
def build(h):
 rows=[]
 for s,d in D.items():
  c=d['close']; r=c.pct_change(); v=r.rolling(20,min_periods=12).std(); m10=c.pct_change(10); m30=c.pct_change(30); m60=c.pct_change(60)
  f=((m10+m30+m60)/(3*(v+1e-8))*(0.5+0.5*(np.sign(m10)+np.sign(m30)+np.sign(m60))/3)).shift(1); fr=c.shift(-h)/c-1
  z=pd.concat([f.rename('f'),fr.rename('fr')],axis=1).dropna(); z['asset']=s; z=z.reset_index(); rows.append(z[['date','f','fr','asset']])
 return pd.concat(rows,ignore_index=True)
def calc(q):
 a=[]; ns=[]
 for _,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: a.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 x=pd.Series(a); return len(x),np.mean(ns),x.mean(),x.mean()/x.std(ddof=1)*np.sqrt(252),(x>0).mean()
q=build(1); print('assets',len(D),'dates',q.date.nunique(),'coverage',len(q)/(len(D)*q.date.nunique()))
for h in [1,5,10,20]: print('horizon',h,calc(build(h)))
for lo,hi in [(2020,2022),(2023,2024),(2025,2027),(2026,2027)]: print('regime',lo,hi,calc(q[(q.date.dt.year>=lo)&(q.date.dt.year<=hi)]))
r=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',r.diff().abs().mean().mean()); q.to_csv('scripts/miner_2_20270325_multihorizon_agreement_signal.csv',index=False)
