import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-05-06')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
 return None
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}; v=fetch('VIX')
if v is None: raise RuntimeError('VIX unavailable')
vp=v.close.astype(float); z=((vp-vp.rolling(60,min_periods=30).mean())/(vp.rolling(60,min_periods=30).std()+1e-12)).shift(1)
rows=[]
for s,d in D.items():
 c=d.close.astype(float); vol=c.pct_change().rolling(20,min_periods=15).std(); base=-(c.pct_change(10))/(vol+1e-12)
 stress=((z.reindex(c.index)-0.25).clip(lower=0,upper=2.0)/2.0)
 rows.append(pd.DataFrame({'date':c.index,'asset':s,'f':(base*stress).values,'fr':c.shift(-1)/c-1}))
q=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
def stats(x):
 vals=[]; ns=[]
 for _,g in x.reset_index(drop=True).groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: vals.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 zz=pd.Series(vals)
 return len(zz),float(np.mean(ns)),float(zz.mean()),float(zz.mean()/zz.std(ddof=1)*np.sqrt(252)),float((zz>0).mean())
print('assets',len(D),'rows',len(q),'dates',q.date.nunique(),'avg_n',q.groupby('date').size().mean(),'coverage',len(q)/(q.date.nunique()*15))
for h in [1,5,10,20]:
 if h==1: x=q
 else:
  rr=[]
  for s,d in D.items(): rr.append(pd.DataFrame({'date':d.index,'asset':s,'f':q[q.asset==s].set_index('date').f.reindex(d.index).values,'fr':d.close.shift(-h)/d.close-1}).reset_index(drop=True))
  x=pd.concat(rr,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
 print('horizon',h,stats(x))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',a,b,stats(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)]))
r=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',r.diff().abs().mean().mean())
q.to_csv('scripts/miner_3_20270506_continuous_stress_reversal_signal.csv',index=False)
