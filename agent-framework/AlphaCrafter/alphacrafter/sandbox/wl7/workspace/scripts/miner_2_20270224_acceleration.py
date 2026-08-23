import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-02-24')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
 return None
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
def make(h):
 out=[]
 for s,d in D.items():
  c=d.close.astype(float); r=c.pct_change(); v=r.rolling(30,min_periods=20).std()
  # acceleration of medium-term trend relative to slow trend, risk scaled and lagged
  f=((c.pct_change(20)-c.pct_change(60)/3)/(v+1e-12)).shift(1); fr=c.shift(-h)/c-1
  out.append(pd.DataFrame({'date':c.index,'asset':s,'f':f,'fr':fr}))
 return pd.concat(out,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
def st(x):
 z=[]; ns=[]
 for _,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: z.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 z=pd.Series(z)
 return len(z),round(float(np.mean(ns)),2),round(float(z.mean()),5),round(float(z.mean()/z.std(ddof=1)*np.sqrt(252)),4),round(float((z>0).mean()),4)
q=make(1); print('assets',len(D),'rows',len(q),'dates',q.date.nunique(),'avg',round(q.groupby('date').size().mean(),2),'coverage',round(len(q)/(q.date.nunique()*len(D)),4))
for h in [1,5,10,20]: print('horizon',h,st(make(h)))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',a,b,st(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)]))
r=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',round(float(r.diff().abs().mean().mean()),5)); q.to_csv('scripts/miner_2_20270224_acceleration_signal.csv',index=False)
