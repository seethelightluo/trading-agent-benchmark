import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-05-17')
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
D={s:load(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
# Residualized, volatility-scaled 30d momentum: asset return relative to same-day cross-asset mean.
px=pd.concat({s:d.close.astype(float) for s,d in D.items()},axis=1).sort_index(); ret=px.pct_change()
r30=px.pct_change(30); mkt=r30.mean(axis=1); vol=ret.rolling(20,min_periods=15).std()
sig=((r30.sub(mkt,axis=0))/vol).shift(1)
fwd=px.shift(-1)/px-1
q=sig.stack().rename('f').to_frame().join(fwd.stack().rename('fr')).reset_index().rename(columns={'level_0':'date','level_1':'asset'}).replace([np.inf,-np.inf],np.nan).dropna()
def stats(x):
 z=[]; ns=[]
 for _,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1:
   v=g.f.corr(g.fr,method='spearman')
   if pd.notna(v): z.append(v); ns.append(len(g))
 z=pd.Series(z); return len(z),float(np.mean(ns)),float(z.mean()),float(z.mean()/z.std(ddof=1)*np.sqrt(252)),float((z>0).mean())
print('assets',len(D),'dates',q.date.nunique(),'coverage',len(q)/(q.date.nunique()*15),'daily',stats(q))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',a,b,stats(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)]))
p=sig.rank(axis=1,pct=True); print('turnover',float(p.diff().abs().mean().mean()))
for h in [1,5,10]:
 fh=px.shift(-h)/px-1; qq=sig.stack().rename('f').to_frame().join(fh.stack().rename('fr')).reset_index().replace([np.inf,-np.inf],np.nan).dropna(); print('horizon',h,'stats',stats(qq))
q.to_csv('scripts/miner_3_20270518_residual_momentum_signal.csv',index=False)
