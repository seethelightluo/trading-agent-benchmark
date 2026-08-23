import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-02-23')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,3000)
   if x is not None and len(x):
    x=x.copy(); x.date=pd.to_datetime(x.date).dt.normalize(); return x.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
 return None
D={s:fetch(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
# Global breadth is computed only from lagged closes, then lagged once for decision safety.
allr=pd.concat({s:d.close.astype(float).pct_change(20) for s,d in D.items()},axis=1)
breadth=(allr>0).mean(axis=1).shift(1)
rows=[]
for s,d in D.items():
 c=d.close.astype(float); r20=c.pct_change(20); vol=c.pct_change().rolling(30,min_periods=20).std()
 # signed momentum, amplified in broad up regimes and inverted in broad down regimes
 regime=(2*breadth.reindex(c.index)-1).clip(-1,1)
 f=(r20/(vol+1e-12)*regime).shift(1)
 fr=c.shift(-1)/c-1
 rows.append(pd.DataFrame({'date':c.index,'asset':s,'f':f,'fr':fr}))
q=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
def stats(x):
 vals=[]; ns=[]
 for _,g in x.reset_index(drop=True).groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:
   vals.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 z=pd.Series(vals)
 return len(z),float(np.mean(ns)),float(z.mean()),float(z.mean()/z.std(ddof=1)*np.sqrt(252)),float((z>0).mean())
print('cutoff',CUT.date(),'assets',len(D),'rows',len(q),'dates',q.date.nunique(),'avg_n',q.groupby('date').size().mean(),'coverage',len(q)/(q.date.nunique()*15))
for h in [1,5,10,20]:
 rr=[]
 for s,d in D.items():
  c=d.close.astype(float); fs=q[q.asset==s].set_index('date').f.reindex(c.index).values
  rr.append(pd.DataFrame({'date':c.index,'asset':s,'f':fs,'fr':c.shift(-h)/c-1}))
 print('horizon',h,stats(pd.concat(rr).replace([np.inf,-np.inf],np.nan).dropna()))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',a,b,stats(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)]))
r=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('rank_turnover',r.diff().abs().mean().mean())
q.to_csv('scripts/miner_3_20270224_breadth_regime_momentum_signal.csv',index=False)
