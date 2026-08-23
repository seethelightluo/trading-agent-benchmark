import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-04-05')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
 return None
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}; rows=[]
# Downside-risk-adjusted medium-term momentum: lagged 20d return divided by downside deviation.
for s,d in D.items():
 c=d.close.astype(float); r=c.pct_change(); down=r.where(r<0,0.0).rolling(30).std(); f=(c.pct_change(20)/(down*np.sqrt(30)+1e-12)).shift(1); fr=c.shift(-1)/c-1
 rows.append(pd.DataFrame({'date':c.index,'asset':s,'f':f,'fr':fr}))
def stats(x):
 z=[]; ns=[]
 for _,g in x.replace([np.inf,-np.inf],np.nan).dropna().groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: z.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 z=pd.Series(z); return len(z),float(np.mean(ns)),float(z.mean()),float(z.mean()/z.std(ddof=1)*np.sqrt(252)),float((z>0).mean())
q=pd.concat(rows,ignore_index=True)
print('assets',len(D),'dates',q.date.nunique(),'rows',len(q),'avg_n',q.dropna().groupby('date').size().mean(),'coverage',len(q.dropna())/(q.date.nunique()*15))
for h in [1,5,10,20]:
 xs=[]
 for s,d in D.items():
  c=d.close.astype(float); r=c.pct_change(); down=r.where(r<0,0.0).rolling(30).std(); f=(c.pct_change(20)/(down*np.sqrt(30)+1e-12)).shift(1); xs.append(pd.DataFrame({'date':c.index,'asset':s,'f':f,'fr':c.shift(-h)/c-1}))
 print('horizon',h,stats(pd.concat(xs,ignore_index=True)))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',a,b,stats(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)]))
r=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',float(r.diff().abs().mean().mean()))
q.to_csv('scripts/miner_2_20270406_downside_adjusted_momentum_signal.csv',index=False); print('signal_artifact scripts/miner_2_20270406_downside_adjusted_momentum_signal.csv')
