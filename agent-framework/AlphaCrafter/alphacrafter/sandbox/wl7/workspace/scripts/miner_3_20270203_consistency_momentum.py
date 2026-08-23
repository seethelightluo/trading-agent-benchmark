import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-02-03')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
 return None
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}; rows=[]
for s,d in D.items():
 c=pd.to_numeric(d.close,errors='coerce'); r=c.pct_change(); mu=r.rolling(20,min_periods=15).mean(); sd=r.rolling(20,min_periods=15).std(); consistency=(r>0).rolling(20,min_periods=15).mean(); f=(mu/(sd+1e-12))*(2*consistency-1)
 rows.append(pd.DataFrame({'date':c.index,'asset':s,'f':f.shift(1),'fr':c.shift(-1)/c-1}))
q=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
def stats(x):
 vals=[]; ns=[]
 for dt,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: vals.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 z=pd.Series(vals); return len(z),round(np.mean(ns),2),round(z.mean(),5),round(z.mean()/z.std(ddof=1)*np.sqrt(252),4),round((z>0).mean(),4)
print('assets',len(D),'rows',len(q),'dates',q.date.nunique(),'avg_n',round(q.groupby('date').size().mean(),2),'coverage',round(len(q)/(q.date.nunique()*len(D)),4))
for h in [1,5,10,20]:
 x=q if h==1 else pd.concat([pd.DataFrame({'date':d.index,'asset':s,'f':q[q.asset==s].set_index('date').f.reindex(d.index).values,'fr':d.close.shift(-h)/d.close-1}) for s,d in D.items()],ignore_index=True).dropna()
 print('horizon',h,stats(x))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',a,b,stats(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)]))
r=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',round(r.diff().abs().mean().mean(),5)); q.to_csv('scripts/miner_3_20270203_consistency_momentum_signal.csv',index=False)
