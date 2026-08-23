import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-03-08')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
 return None
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
# Candidate: downside-risk-adjusted medium trend. Reward returns, penalize only harmful
# negative daily moves; lagged one session to avoid look-ahead.
def build(h):
 out=[]
 for s,d in D.items():
  c=d.close; r=c.pct_change(); down=r.where(r<0,0.0).pow(2).rolling(20,min_periods=15).mean().pow(.5)
  sig=(c.pct_change(15)/(down*np.sqrt(20)+1e-12)).shift(1)
  fwd=c.shift(-h)/c-1
  out.append(pd.DataFrame({'date':c.index,'asset':s,'signal':sig,'forward_return':fwd}))
 return pd.concat(out,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
def calc(q):
 vals=[]; ns=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.signal.nunique()>1 and g.forward_return.nunique()>1:
   vals.append(g.signal.corr(g.forward_return,method='spearman')); ns.append(len(g))
 x=pd.Series(vals); return len(x),np.mean(ns),x.mean(),x.mean()/x.std(ddof=1)*np.sqrt(252), (x>0).mean()
q=build(1); print('assets',len(D),'dates',q.date.nunique(),'coverage',len(q)/(q.date.nunique()*len(D)),'avg_n',q.groupby('date').size().mean())
for h in [1,5,10,20]: print('horizon',h,calc(build(h)))
for lo,hi in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',lo,hi,calc(q[(q.date.dt.year>=lo)&(q.date.dt.year<=hi)]))
p=q.pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True); print('turnover',p.diff().abs().mean().mean())
q.to_csv('scripts/miner_3_20270309_downside_adjusted_trend_signal.csv',index=False)
