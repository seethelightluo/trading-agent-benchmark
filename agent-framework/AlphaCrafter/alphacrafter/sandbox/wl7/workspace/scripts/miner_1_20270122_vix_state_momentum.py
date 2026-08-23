import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-01-21')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
 return None
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
vix=pd.read_csv('../persistent/index_data/VIX.csv'); vix.date=pd.to_datetime(vix.date).dt.normalize(); vix=vix.set_index('date').sort_index().loc[:CUT].close.astype(float)
# Volatility-state momentum: 10-day return normalized by 30-day volatility, with
# a continuous dampener when VIX is above its 60-day median. Inputs lagged one day.
def make(h):
 out=[]
 for s,d in D.items():
  c=d.close.astype(float); r=c.pct_change(); vol=r.rolling(30,min_periods=20).std()
  vm=vix.reindex(c.index).ffill(); med=vm.rolling(60,min_periods=40).median()
  f=(c.pct_change(10)/(vol+1e-12) * (med/(vm+1e-12)).clip(0.35,1.5)).shift(1)
  fr=c.shift(-h)/c-1
  out.append(pd.DataFrame({'date':c.index,'asset':s,'f':f,'fr':fr}))
 return pd.concat(out,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
def stats(q):
 z=[]; ns=[]
 for _,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:
   z.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 z=pd.Series(z); return {'dates':len(z),'avg_n':float(np.mean(ns)),'ic':float(z.mean()),'icir':float(z.mean()/z.std(ddof=1)*np.sqrt(252)),'hit':float((z>0).mean())}
q=make(1); print('assets',len(D),'rows',len(q),'date_span',q.date.min(),q.date.max())
for h in [1,5,10,20]: print('horizon',h,stats(make(h)))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',a,b,stats(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)]))
r=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',float(r.diff().abs().mean().mean()),'coverage',len(q)/(q.date.nunique()*15))
q.to_csv('scripts/miner_1_20270122_vix_state_momentum_signal.csv',index=False)
