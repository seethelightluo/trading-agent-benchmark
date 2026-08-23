import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUTOFF=pd.Timestamp('2026-10-30')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,3000)
   if x is not None and len(x):
    x=x.copy(); x.date=pd.to_datetime(x.date).dt.normalize(); return x.drop_duplicates('date').set_index('date').sort_index()
  except Exception: pass
D={s:fetch(s) for s in U}; D={s:d.loc[d.index<=CUTOFF] for s,d in D.items() if d is not None}
def factor(d,h):
 c=d.close.replace(0,np.nan); r=c.pct_change(); down=(-r.clip(upper=0)).pow(2).rolling(20,min_periods=15).mean().pow(.5)
 f=(c.pct_change(20)/(down+1e-8)).shift(1); fr=c.shift(-h)/c-1
 return pd.DataFrame({'f':f,'fr':fr}).replace([np.inf,-np.inf],np.nan).dropna()
def calc(h,lo=None,hi=None):
 rows=[]
 for s,d in D.items():
  q=factor(d,h)
  if lo: q=q.loc[q.index>=pd.Timestamp(lo)]
  if hi: q=q.loc[q.index<=pd.Timestamp(hi)]
  q['asset']=s; rows.append(q.reset_index())
 q=pd.concat(rows,ignore_index=True); ic=[]; ns=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: ic.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 a=pd.Series(ic); return len(a),round(float(np.mean(ns)),2),float(a.mean()),float(a.mean()/a.std(ddof=1)*np.sqrt(252)),float((a>0).mean()),q.date.nunique()
print('assets',len(D),'cutoff',CUTOFF.date(),'date span',min(d.index.min() for d in D.values()).date(),max(d.index.max() for d in D.values()).date())
for h in [1,5,10,20]: print('horizon',h,calc(h))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-10-30')]: print('regime',lo[:4]+'-'+hi[:4],calc(1,lo,hi))
R=[]
for s,d in D.items(): R.append(factor(d,1).f.rename(s))
r=pd.concat(R,axis=1).rank(axis=1,pct=True); print('turnover',float(r.diff().abs().mean(axis=1).mean()),'rank_dates',len(r),'coverage',float(r.notna().mean().mean()))
out=pd.concat([factor(d,1).f.rename(s) for s,d in D.items()],axis=1); out.to_csv('scripts/miner_2_20261102_downside_trend_signal.csv'); print('signal_artifact scripts/miner_2_20261102_downside_trend_signal.csv')
