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
D={s:fetch(s) for s in U}; D={s:d.loc[:'2026-11-13'] for s,d in D.items() if d is not None}
def calc(h):
 q=[]
 for s,d in D.items():
  c=d.close.replace([np.inf,-np.inf],np.nan); r=c.pct_change(); v=r.rolling(20,min_periods=10).std(); dn=r.where(r<0,0).rolling(20,min_periods=10).std(); z=(dn-dn.rolling(60,min_periods=20).median())/(dn.rolling(60,min_periods=20).std()+1e-8)
  f=(-c.pct_change(2)/(v+1e-8))*(1+0.5*z.clip(0,2)); fr=c.shift(-h)/c-1
  q.append(pd.DataFrame({'f':f,'fr':fr,'asset':s}))
 q=pd.concat(q).dropna().reset_index(); vals=[]; ns=[]
 for _,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: vals.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 a=pd.Series(vals); return len(a),float(np.mean(ns)),float(a.mean()),float(a.mean()/a.std(ddof=1)*np.sqrt(252)),float((a>0).mean())
print('assets',len(D),'cutoff','2026-11-13')
for h in [1,5,10,20]: print('horizon',h,'dates avg_names IC ICIR hit',calc(h))
for y in range(2020,2027):
 # replicate yearly from 1d
 q=[]
 for s,d in D.items():
  c=d.close;r=c.pct_change();v=r.rolling(20,min_periods=10).std();dn=r.where(r<0,0).rolling(20,min_periods=10).std();z=(dn-dn.rolling(60,min_periods=20).median())/(dn.rolling(60,min_periods=20).std()+1e-8);f=(-c.pct_change(2)/(v+1e-8))*(1+.5*z.clip(0,2));q.append(pd.DataFrame({'f':f,'fr':c.shift(-1)/c-1}))
 q=pd.concat(q).dropna().reset_index(); a=[]
 for dt,g in q[q.date.dt.year==y].groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:a.append(g.f.corr(g.fr,method='spearman'))
 a=pd.Series(a);print('regime',y,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1)*np.sqrt(252) if len(a)>1 else np.nan)
