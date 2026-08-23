import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-06-08')
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,3000)
   if x is not None and len(x):
    x=x.copy(); x.date=pd.to_datetime(x.date).dt.normalize(); return x.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
D={s:load(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
def build(h):
 out=[]
 for s,d in D.items():
  c=d.close.astype(float); r=c.pct_change(); mom=c/c.shift(20)-1
  vol=r.rolling(20,min_periods=15).std(); consistency=r.rolling(20,min_periods=15).apply(lambda x: np.mean(x>0),raw=True)
  # smooth trend quality: momentum divided by volatility and rewarded for directional breadth
  f=(mom/(vol+1e-12))*(0.5+consistency)
  out.append(pd.DataFrame({'date':c.index,'asset':s,'f':f.shift(1),'fr':c.shift(-h)/c-1}))
 return pd.concat(out,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
def stats(q):
 z=[]; ns=[]
 for _,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: z.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 z=pd.Series(z).dropna(); return len(z),np.mean(ns),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252),(z>0).mean()
q=build(10)
print('assets',len(D),'dates',q.date.nunique(),'avg_n',q.groupby('date').size().mean(),'coverage',len(q)/(q.date.nunique()*15)); print('10d',stats(q))
for h in [1,5,10,20]: print('horizon',h,stats(build(h)))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',a,b,stats(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)]))
r=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',r.diff().abs().mean().mean())
q.to_csv('scripts/miner_1_20270609_quality_momentum_signal.csv',index=False)
