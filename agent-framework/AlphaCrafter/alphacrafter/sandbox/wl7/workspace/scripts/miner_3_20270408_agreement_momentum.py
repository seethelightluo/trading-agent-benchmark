import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-04-07')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
rets=pd.DataFrame({s:d.close.astype(float).pct_change(10) for s,d in D.items()}); breadth=(rets>0).mean(axis=1)-0.5

def make(h):
 rows=[]
 for s,d in D.items():
  c=d.close.astype(float); r10=c.pct_change(10); vol=c.pct_change().rolling(30,min_periods=20).std()
  f=(r10/(vol+1e-12)*breadth.reindex(c.index)).shift(1); fr=c.shift(-h)/c-1
  rows.append(pd.DataFrame({'date':c.index,'asset':s,'f':f.values,'fr':fr.values}))
 return pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
def stats(x):
 z=[]; ns=[]
 for dt,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: z.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 z=pd.Series(z); return len(z),float(np.mean(ns)),float(z.mean()),float(z.mean()/z.std(ddof=1)*np.sqrt(252)),float((z>0).mean())
q=make(1); print('assets',len(D),'rows',len(q),'dates',q.date.nunique(),'avg',q.groupby('date').size().mean())
for h in [1,5,10,20]: print('horizon',h,stats(make(h)))
for lo,hi in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',lo,hi,stats(q[(q.date.dt.year>=lo)&(q.date.dt.year<=hi)]))
r=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',r.diff().abs().mean().mean(),'coverage',len(q)/(q.date.nunique()*15))
q.to_csv('scripts/miner_3_20270408_agreement_momentum_signal.csv',index=False)
