import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-05-20')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
 return None
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
P=pd.concat({s:d.close for s,d in D.items()},axis=1).sort_index(); R=P.pct_change()
# Cross-asset residual reversal: lagged 10-session asset return relative to contemporaneous cross-sectional median,
# scaled by each asset's 20-session volatility. Designed to exploit short-horizon mean reversion after idiosyncratic moves.
rows=[]
for s,d in D.items():
 c=d.close.astype(float); r=c.pct_change(); vol=r.rolling(20,min_periods=15).std()
 common=(R[s]-R.median(axis=1)).rolling(10,min_periods=8).sum().reindex(c.index)
 f=-(common/(vol*np.sqrt(10)+1e-12)).shift(1)
 for h in [1,5,10,20]:
  rows.append(pd.DataFrame({'date':c.index,'asset':s,'f':f,'fr':c.shift(-h)/c-1,'h':h}))
q=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
def stats(x):
 z=[]; ns=[]
 for _,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: z.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 z=pd.Series(z)
 return len(z),float(np.mean(ns)),float(z.mean()),float(z.mean()/z.std(ddof=1)*np.sqrt(252)) if len(z)>1 else np.nan,float((z>0).mean())
print('assets',len(D),'dates',q[q.h==1].date.nunique(),'avg_n',q[q.h==1].groupby('date').size().mean(),'coverage',len(q[q.h==1])/(q[q.h==1].date.nunique()*15))
for h in [1,5,10,20]:
 x=q[q.h==h]; print('horizon',h,stats(x))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',a,b,stats(q[(q.h==1)&(q.date.dt.year>=a)&(q.date.dt.year<=b)]))
r=q[q.h==1].pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',r.diff().abs().mean().mean())
q[q.h==1].to_csv('scripts/miner_2_20270521_residual_reversal_signal.csv',index=False)
