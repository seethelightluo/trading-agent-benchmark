import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-04-13')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
def make(h):
 out=[]
 for s,d in D.items():
  c=d.close.astype(float); r=c.pct_change(); v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std()
  # Contrarian volatility expansion: assets with rising short volatility have recently sold off,
  # and cross-asset reversal can reward the overshoot. Lag one completed session.
  f=(v20/(v60+1e-8)).shift(1); fr=c.shift(-h)/c-1
  out.append(pd.DataFrame({'date':c.index,'asset':s,'f':f.values,'fr':fr.values}))
 return pd.concat(out,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
def stats(x):
 z=[]; nn=[]
 for _,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: z.append(g.f.corr(g.fr,method='spearman')); nn.append(len(g))
 z=pd.Series(z); return len(z),float(np.mean(nn)),float(z.mean()),float(z.mean()/z.std(ddof=1)*np.sqrt(252)),float((z>0).mean())
q=make(1); print('assets',len(D),'rows',len(q),'dates',q.date.nunique(),'avg_instruments',q.groupby('date').size().mean())
for h in [1,5,10,20]: print('horizon',h,'dates avgN IC ICIR hit',stats(make(h)))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',a,b,stats(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)]))
r=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('rank_turnover',r.diff().abs().mean().mean(),'coverage',len(q)/(q.date.nunique()*15)); q.to_csv('scripts/miner_1_20270414_vol_expansion_reversal_signal.csv',index=False)
