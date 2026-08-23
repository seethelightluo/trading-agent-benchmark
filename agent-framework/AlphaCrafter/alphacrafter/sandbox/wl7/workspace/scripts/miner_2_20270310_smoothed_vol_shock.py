import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-03-10')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
 return None
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}; rows=[]
for s,d in D.items():
 c=d.close.astype(float); r=c.pct_change(); v10=r.rolling(10,min_periods=7).std(); v60=r.rolling(60,min_periods=30).std()
 base=(-(c.pct_change(3))/(v10+1e-12)*(v10/(v60+1e-12))).shift(1); f=base.rolling(3,min_periods=2).mean()
 for h in [1,5,10]:
  fr=c.shift(-h)/c-1
  for dt,x,y in zip(c.index,f,fr): rows.append({'date':dt,'asset':s,'f':x,'h':h,'fr':y})
q=pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan)
def stats(x):
 z=[]; ns=[]
 for _,g in x[['date','f','fr']].dropna().groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: z.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 z=pd.Series(z); return len(z),np.mean(ns),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252), (z>0).mean()
print('assets',len(D),'dates',q.date.nunique(),'avg_instruments',q.groupby('date').f.count().mean(),'coverage',q.f.notna().mean())
for h in [1,5,10]: print('horizon',h,stats(q[q.h==h]))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',a,b,stats(q[(q.h==1)&(q.date.dt.year>=a)&(q.date.dt.year<=b)]))
one=q[q.h==1]; r=one.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',r.diff().abs().mean().mean())
q.to_csv('scripts/miner_2_20270310_smoothed_vol_shock_signal.csv',index=False)
