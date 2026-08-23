import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUTOFF=pd.Timestamp('2026-12-11')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUTOFF]
  except Exception: pass
 return None
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
def make(d,h):
 c=d.close.replace(0,np.nan); r=c.pct_change(); loss=(-r.clip(upper=0)).rolling(5,min_periods=4).sum(); gain=r.clip(lower=0).rolling(5,min_periods=4).sum(); dv=(-r.clip(upper=0)).pow(2).rolling(20,min_periods=12).mean().pow(.5); f=((loss-gain)/(dv+1e-8)).shift(1); fr=c.shift(-h)/c-1
 return pd.DataFrame({'f':f,'fr':fr}).replace([np.inf,-np.inf],np.nan).dropna()
def calc(h,lo=None,hi=None):
 rows=[]
 for s,d in D.items():
  q=make(d,h); mask=np.ones(len(q),dtype=bool); mask &= (q.index>=pd.Timestamp(lo)) if lo else True; mask &= (q.index<=pd.Timestamp(hi)) if hi else True; q=q.loc[mask]; q['asset']=s; rows.append(q.reset_index())
 q=pd.concat(rows); vals=[]; ns=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: vals.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 a=pd.Series(vals); return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(252),(a>0).mean(),q.date.nunique()
print('assets',len(D),'lengths',min(map(len,D.values())),max(map(len,D.values())))
for h in [1,5,10,20]: print('horizon',h,calc(h))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-12-11')]: print('regime',lo[:4]+'-'+hi[:4],calc(1,lo,hi))
rows=[make(d,1).f.rename(s) for s,d in D.items()]; x=pd.concat(rows,axis=1); ranks=x.rank(axis=1,pct=True); print('turnover',ranks.diff().abs().mean(axis=1).mean(),'rank_dates',len(ranks),'period',ranks.index.min().date(),ranks.index.max().date())
# signal artifact for provenance
ranks.to_csv('scripts/miner_3_20261214_persistent_loss_reversal_signal.csv')
