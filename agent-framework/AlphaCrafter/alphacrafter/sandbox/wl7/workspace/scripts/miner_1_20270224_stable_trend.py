import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-02-23')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
 return None
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
# Stable trend: direction-consistent multi-horizon returns, scaled by downside risk.
def factor(h):
 out=[]
 for s,d in D.items():
  c=d.close.astype(float); r=c.pct_change();
  r5=c.pct_change(5); r20=c.pct_change(20); r60=c.pct_change(60)
  down=r.where(r<0,0).rolling(60,min_periods=40).std()
  f=((r5/5+r20/20+r60/60)/3)/(down+1e-5)
  out.append(pd.DataFrame({'date':c.index,'asset':s,'f':f.shift(1),'fr':c.shift(-h)/c-1}))
 return pd.concat(out,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
def st(x):
 z=[]; ns=[]
 for _,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: z.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 z=pd.Series(z)
 return len(z),float(np.mean(ns)),float(z.mean()),float(z.mean()/z.std(ddof=1)*np.sqrt(252)),float((z>0).mean())
q=factor(1); print('assets',len(D),'rows',len(q),'dates',q.date.nunique(),'avg',q.groupby('date').size().mean())
for h in [1,5,10,20]: print('horizon',h,st(factor(h)))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',a,b,st(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)]))
r=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True)
print('turnover',float(r.diff().abs().mean().mean()),'coverage',len(q)/(q.date.nunique()*15)); q.to_csv('scripts/miner_1_20270224_stable_trend_signal.csv',index=False)
