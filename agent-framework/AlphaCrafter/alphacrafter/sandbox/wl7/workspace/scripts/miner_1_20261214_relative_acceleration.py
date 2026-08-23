import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,3000)
   if x is not None and len(x):
    x=x.copy();x.date=pd.to_datetime(x.date).dt.normalize();return x.drop_duplicates('date').set_index('date').sort_index()
  except Exception: pass
D={s:get(s) for s in U};D={s:x for s,x in D.items() if x is not None}
# Relative acceleration: recent 5d return relative to its prior 15d trend, normalized by recent volatility.
def fac(d,h):
 c=d.close;r=c.pct_change();v=r.rolling(20,min_periods=15).std()
 acceleration=(c/c.shift(5)-1)-((c.shift(5)/c.shift(20)-1)/3.0)
 f=(acceleration/(v*np.sqrt(5)+1e-8)).shift(1)
 fr=c.shift(-h)/c-1
 return pd.DataFrame({'f':f,'fr':fr}).dropna()
def calc(h,lo=None,hi=None):
 rows=[]
 for s,d in D.items():
  q=fac(d,h)
  if lo is not None:q=q[(q.index.year>=lo)&(q.index.year<=hi)]
  q['asset']=s;rows.append(q.reset_index())
 q=pd.concat(rows);ics=[];ns=[]
 for _,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:
   z=g.f.corr(g.fr,method='spearman')
   if pd.notna(z):ics.append(z);ns.append(len(g))
 a=pd.Series(ics);return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(252),(a>0).mean()
print('assets',len(D),'dates',min(x.index.min() for x in D.values()),max(x.index.max() for x in D.values()))
for h in [1,5,10,20]:print('horizon',h,'dates avg_names IC ICIR hit',calc(h))
for x in [(2020,2022),(2023,2024),(2025,2026)]:print('regime',x,calc(1,*x))
rows=[]
for s,d in D.items():
 q=fac(d,1);q['asset']=s;rows.append(q.reset_index())
q=pd.concat(rows);p=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True)
print('valid_dates',len(p),'coverage',p.notna().mean().mean(),'turnover',p.diff().abs().mean().mean())
