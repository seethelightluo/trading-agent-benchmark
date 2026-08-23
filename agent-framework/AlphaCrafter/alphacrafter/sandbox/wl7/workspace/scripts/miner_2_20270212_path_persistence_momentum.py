import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-02-12')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
 return None
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
rows=[]
for s,d in D.items():
 c=d.close.astype(float); ret=c.pct_change(); path=ret.abs().rolling(20,min_periods=15).sum()
 f=(c.pct_change(15)/(path+1e-12)).shift(1); fr=c.shift(-5)/c-1
 rows.append(pd.DataFrame({'date':c.index,'asset':s,'f':f,'fr':fr}))
q=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
def stats(x):
 z=[]; ns=[]
 for _,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: z.append(g.f.corr(g.fr)); ns.append(len(g))
 z=pd.Series(z); return len(z),float(np.mean(ns)),float(z.mean()),float(z.mean()/z.std(ddof=1)),float((z>0).mean())
print('assets',len(D),'rows',len(q),'dates',q.date.nunique(),'avg_n',q.groupby('date').size().mean(),'coverage',len(q)/(q.date.nunique()*15)); print('horizon5',stats(q))
for h in [1,10,20]:
 rr=[]
 for s,d in D.items(): rr.append(pd.DataFrame({'date':d.index,'asset':s,'f':q[q.asset==s].set_index('date').f.reindex(d.index).values,'fr':(d.close.shift(-h)/d.close-1).values}))
 print('horizon',h,stats(pd.concat(rr).replace([np.inf,-np.inf],np.nan).dropna()))
r=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',r.diff().abs().mean().mean()); print('period',q.date.min(),q.date.max()); q.to_csv('scripts/miner_2_20270212_path_persistence_momentum_signal.csv',index=False)
