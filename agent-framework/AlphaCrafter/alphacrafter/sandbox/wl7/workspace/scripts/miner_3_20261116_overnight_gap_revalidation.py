import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT='2026-11-13'
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy();d.date=pd.to_datetime(d.date).dt.normalize();return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
 return None
D={s:fetch(s) for s in U};D={s:d for s,d in D.items() if d is not None}
def calc(h):
 rows=[]
 for s,d in D.items():
  c=d.close.replace([np.inf,-np.inf],np.nan); f=-(d.open/d.close.shift(1)-1); fr=c.shift(-h)/c-1
  rows.append(pd.DataFrame({'f':f,'fr':fr,'asset':s}))
 q=pd.concat(rows).dropna().reset_index(); vals=[]; ns=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: vals.append(g.f.corr(g.fr,method='spearman'));ns.append(len(g))
 a=pd.Series(vals);return len(a),float(np.mean(ns)),float(a.mean()),float(a.mean()/a.std(ddof=1)*np.sqrt(252)),float((a>0).mean()),float(a.std(ddof=1))
print('assets',len(D),'cutoff',CUT)
for h in [1,5,10,20]:print('horizon',h,'dates avg_names IC ICIR hit std',calc(h))
# annual subperiods for the admission horizon
rows=[]
for s,d in D.items():
 c=d.close; rows.append(pd.DataFrame({'f':-(d.open/d.close.shift(1)-1),'fr':c.shift(-1)/c-1,'asset':s}))
q=pd.concat(rows).dropna().reset_index()
for y in range(2020,2027):
 a=[]
 for _,g in q[q.date.dt.year==y].groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:a.append(g.f.corr(g.fr,method='spearman'))
 a=pd.Series(a);print('regime',y,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1)*np.sqrt(252) if len(a)>1 else np.nan)
p=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True)
print('coverage',float(p.notna().mean().mean()),'turnover',float(p.diff().abs().mean().mean()),'date_count',len(p),'avg_assets',float(p.notna().sum(axis=1).mean()))
