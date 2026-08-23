import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2026-10-07')
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d[d.date<=CUT].drop_duplicates('date').set_index('date').sort_index()
  except Exception: pass
D={s:get(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
def series(d,h=1):
 o=d.open.replace(0,np.nan); c=d.close; r=(d.high-d.low).replace(0,np.nan)
 # lagged candle pressure: close location weighted by body and normalized by recent range
 clv=(2*c-d.high-d.low)/r
 body=(c-o)/o
 vol=r/o
 f=(clv*body/(vol.rolling(20,min_periods=10).median()+1e-8)).ewm(span=3,min_periods=3,adjust=False).mean().shift(1)
 fr=c.shift(-h)/c-1
 return pd.DataFrame({'f':f,'fr':fr}).dropna()
def calc(h,lo=None,hi=None):
 z=[]
 for s,d in D.items():
  q=series(d,h); q['asset']=s; z.append(q.reset_index())
 q=pd.concat(z); q=q[(q.date>=lo)&(q.date<=hi)] if lo is not None else q
 vals=[]; ns=[]; turns=[]
 for _,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:
   vals.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 a=pd.Series(vals); return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(252),(a>0).mean()
print('assets',len(D),'date_range',min(d.index.min() for d in D.values()),max(d.index.max() for d in D.values()))
for h in [1,5,10,20]: print('horizon',h,'dates avg_names IC ICIR hit',calc(h))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-10-07')]: print('regime',lo,hi,calc(1,pd.Timestamp(lo),pd.Timestamp(hi)))
# rank turnover on consecutive common dates
z=[]
for s,d in D.items():
 q=series(d,1); q['asset']=s; z.append(q.reset_index())
q=pd.concat(z).pivot(index='date',columns='asset',values='f'); ranks=q.rank(pct=True); turns=(ranks-ranks.shift(1)).abs().mean(axis=1).dropna()
print('turnover_proxy',turns.mean(),'coverage',q.notna().mean().mean(),'dates',len(q))
