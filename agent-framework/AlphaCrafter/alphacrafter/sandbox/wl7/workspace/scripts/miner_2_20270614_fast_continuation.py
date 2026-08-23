import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-06-13')
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
D={s:get(s) for s in U}; D={s:d for s,d in D.items() if d is not None}; out=[]
for s,d in D.items():
 c=d.close.astype(float); ret=c.pct_change(); v=ret.rolling(15,min_periods=10).std()*np.sqrt(3)
 # Fast continuation only when medium trend agrees; lagged one day.
 f=(c.pct_change(3)/(v+1e-12)*np.where(c.pct_change(30)>=0,1.,-0.25)).shift(1)
 out.append(pd.DataFrame({'date':c.index,'asset':s,'f':f}))
P=pd.concat(out,ignore_index=True); rr=[]
for s,d in D.items():
 c=d.close.astype(float); z=P[P.asset==s].set_index('date').f.reindex(c.index)
 rr.append(pd.DataFrame({'date':c.index,'asset':s,'f':z.values,'fr':c.shift(-1).values/c.values-1}))
x=pd.concat(rr).replace([np.inf,-np.inf],np.nan).dropna(); vals=[]; ns=[]
for dt,g in x.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: vals.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
z=pd.Series(vals); print('assets',len(D),'dates',len(z),'avg_n',np.mean(ns),'ic',z.mean(),'icir',z.mean()/z.std(ddof=1)*np.sqrt(252),'hit',(z>0).mean(),'coverage',len(x)/(x.date.nunique()*len(U)))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]:
 q=x[x.date.dt.year.between(a,b)]; v=[]
 for _,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:v.append(g.f.corr(g.fr,method='spearman'))
 print('regime',a,b,len(v),np.mean(v))
r=P.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',r.diff().abs().mean().mean())
P.to_csv('scripts/miner_2_20270614_fast_continuation_signal.csv',index=False)
