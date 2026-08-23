import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-04-07')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
# Candidate: residual medium-term momentum: asset 20d return less rolling beta to SPX times SPX return,
# divided by asset 20d realized volatility. All inputs lagged one day.
prices=pd.concat({s:d.close.astype(float) for s,d in D.items()},axis=1).sort_index(); ret=prices.pct_change()
bench=ret['SPX']
rows=[]
for s in D:
 r=ret[s]; beta=r.rolling(60,min_periods=30).cov(bench)/bench.rolling(60,min_periods=30).var()
 resid=(r.rolling(20).sum()-beta*bench.rolling(20).sum()).shift(1)
 vol=r.rolling(20).std().shift(1)*np.sqrt(20)
 f=resid/(vol+1e-12); fr=prices[s].shift(-1)/prices[s]-1
 rows.append(pd.DataFrame({'date':prices.index,'asset':s,'f':f,'fr':fr}))
q=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
def stats(x):
 z=[]; ns=[]
 for dt,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: z.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 z=pd.Series(z)
 return len(z),float(np.mean(ns)),float(z.mean()),float(z.mean()/z.std(ddof=1)*np.sqrt(252)),float((z>0).mean())
print('assets',len(D),'rows',len(q),'dates',q.date.nunique(),'avg',q.groupby('date').size().mean())
for h in [1,5,10,20]:
 # recompute forward horizon from prices, retaining f
 x=[]
 for s in D:
  fr=prices[s].shift(-h)/prices[s]-1
  x.append(pd.DataFrame({'date':prices.index,'asset':s,'f':q.loc[q.asset.eq(s)].set_index('date').f,'fr':fr}))
 x=pd.concat(x).reset_index(drop=True).replace([np.inf,-np.inf],np.nan).dropna(); print('horizon',h,stats(x))
for lo,hi in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',lo,hi,stats(q[(q.date.dt.year>=lo)&(q.date.dt.year<=hi)]))
r=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',r.diff().abs().mean().mean(),'coverage',len(q)/(q.date.nunique()*15))
q.to_csv('scripts/miner_2_20270408_residual_momentum_signal.csv',index=False)
