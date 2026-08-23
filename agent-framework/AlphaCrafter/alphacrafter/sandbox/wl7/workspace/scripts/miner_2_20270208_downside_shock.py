import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-02-07')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize()
    return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
 return None
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
def factor(d):
 c=d.close.replace(0,np.nan); r=c.pct_change()
 down=(-r.clip(upper=0)).rolling(3,min_periods=3).sum()
 up=r.clip(lower=0).rolling(10,min_periods=8).sum()
 dv=(-r.clip(upper=0)).pow(2).rolling(20,min_periods=12).mean().pow(.5)
 return ((down/(dv+1e-8))-0.25*(up/(dv+1e-8))).shift(1), c

def run(h,lo=None,hi=None):
 rows=[]
 for s,d in D.items():
  f,c=factor(d); q=pd.DataFrame({'f':f,'fr':c.shift(-h)/c-1}).replace([np.inf,-np.inf],np.nan).dropna()
  if lo:q=q.loc[q.index>=pd.Timestamp(lo)]
  if hi:q=q.loc[q.index<=pd.Timestamp(hi)]
  q['asset']=s;rows.append(q.reset_index())
 q=pd.concat(rows,ignore_index=True); vals=[]; ns=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:
   vals.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 a=pd.Series(vals); return {'observations':len(a),'avg_n':np.mean(ns),'ic':a.mean(),'icir':a.mean()/a.std(ddof=1)*np.sqrt(252),'hit':(a>0).mean(),'dates':q.date.nunique()}
print('assets',len(D),'dates',min(d.index.min() for d in D.values()),max(d.index.max() for d in D.values()))
for h in [1,2,5,10,20]: print('h',h,run(h))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-02-07')]: print('regime',lo[:4]+'-'+hi[:4],run(1,lo,hi))
r=pd.concat([factor(d)[0].rename(s) for s,d in D.items()],axis=1).rank(axis=1,pct=True)
print('coverage',r.notna().mean().mean(),'turnover',r.diff().abs().mean(axis=1).mean(),'rank_dates',len(r))
r.to_csv('scripts/miner_2_20270208_downside_shock_signal.csv')
