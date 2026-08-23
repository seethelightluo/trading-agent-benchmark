import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-04-20')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
 return None
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
P=pd.concat({s:d.close.astype(float) for s,d in D.items()},axis=1).sort_index(); R=P.pct_change(); rows=[]
# Interpretable efficiency trend: directional net return divided by travelled path and risk.
for s in D:
 r=R[s]; net=r.rolling(20,min_periods=15).sum(); path=r.abs().rolling(20,min_periods=15).sum(); vol=r.rolling(40,min_periods=25).std()*np.sqrt(20)
 f=(net/(path+1e-12)/(vol+1e-12)).shift(1); fr=P[s].shift(-1)/P[s]-1
 rows.append(pd.DataFrame({'date':P.index,'asset':s,'f':f,'fr':fr}))
q=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
def stats(x):
 vals=[]; ns=[]
 for _,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: vals.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 z=pd.Series(vals); return len(z),np.mean(ns),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252),(z>0).mean()
print('assets',len(D),'rows',len(q),'dates',q.date.nunique(),'avg_n',q.groupby('date').size().mean(),'coverage',len(q)/(q.date.nunique()*15))
for h in [1,5,10,20]:
 xs=[]
 for s,d in D.items():
  sig=q[q.asset==s].set_index('date').f.reindex(d.index).values
  xs.append(pd.DataFrame({'date':d.index,'asset':s,'f':sig,'fr':(d.close.shift(-h)/d.close-1).values}))
 print('horizon',h,stats(pd.concat(xs).replace([np.inf,-np.inf],np.nan).dropna()))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',a,b,stats(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)]))
r=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',r.diff().abs().mean().mean())
q.to_csv('scripts/miner_2_20270421_efficiency_trend_signal.csv',index=False); print('signal_artifact scripts/miner_2_20270421_efficiency_trend_signal.csv')
