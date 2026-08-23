import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-08-31')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
 return None
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
vix=fetch('VIX')
if vix is None: raise RuntimeError('VIX unavailable')
vp=vix.close.astype(float); vz=((vp-vp.rolling(60,min_periods=30).mean())/(vp.rolling(60,min_periods=30).std()+1e-12)).shift(1)
rows=[]
for s,d in D.items():
 c=d.close.astype(float); r=c.pct_change(); vol=r.rolling(20,min_periods=15).std()
 # Contrarian score, amplified by lagged VIX stress but never using current/future observations.
 base=-(c.pct_change(10))/(vol+1e-12)
 stress=1+0.75*vz.clip(lower=0,upper=3).reindex(c.index).fillna(0)
 sig=base*stress
 fr=c.shift(-1)/c-1
 rows.append(pd.DataFrame({'date':c.index,'asset':s,'f':sig.values,'fr':fr.values}))
q=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
def stats(x):
 z=[]; ns=[]
 for _,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: z.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 z=pd.Series(z)
 return len(z),float(np.mean(ns)),float(z.mean()),float(z.mean()/z.std(ddof=1)*np.sqrt(252)),float((z>0).mean())
print('assets',len(D),'dates',q.date.nunique(),'avg_n',q.groupby('date').size().mean(),'coverage',len(q)/(q.date.nunique()*15))
for h in [1,5,10,20]:
 rr=[]
 for s,d in D.items(): rr.append(pd.DataFrame({'date':d.index,'asset':s,'f':q[q.asset==s].set_index('date').f.reindex(d.index).values,'fr':(d.close.shift(-h)/d.close-1).values}))
 print('horizon',h,stats(pd.concat(rr).replace([np.inf,-np.inf],np.nan).dropna()))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',a,b,stats(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)]))
r=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',r.diff().abs().mean().mean())
q.to_csv('scripts/miner_1_20270901_stress_adaptive_reversal_signal.csv',index=False)
