import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-03-21')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
 return None
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
def panel(h):
 rows=[]
 for s,d in D.items():
  c=d.close.astype(float); r=c.pct_change()
  vol=r.rolling(20,min_periods=15).std(); eff=r.rolling(20,min_periods=15).sum().abs()/(r.abs().rolling(20,min_periods=15).sum()+1e-12)
  # lagged, directional trend strength: signed momentum, volatility adjusted and efficiency weighted
  f=(r.rolling(15,min_periods=12).sum()/(vol*np.sqrt(20)+1e-12)*(0.5+eff)).shift(1)
  fr=c.shift(-h)/c-1
  rows.append(pd.DataFrame({'date':c.index,'asset':s,'f':f,'fr':fr}))
 return pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
def stats(x):
 z=[]; ns=[]
 for dt,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: z.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 z=pd.Series(z); return len(z),round(np.mean(ns),2),round(z.mean(),5),round(z.mean()/z.std(ddof=1)*np.sqrt(252),4),round((z>0).mean(),4)
print('assets',len(D))
q=panel(1); print('rows',len(q),'dates',q.date.nunique(),'avg instruments',round(q.groupby('date').size().mean(),2))
for h in [1,2,5,10,20]: print('horizon',h,stats(panel(h)))
for lo,hi in [(2020,2022),(2023,2024),(2025,2027),(2026,2027)]: print('regime',lo,hi,stats(q[(q.date.dt.year>=lo)&(q.date.dt.year<=hi)]))
r=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',round(r.diff().abs().mean().mean(),4),'coverage',round(len(q)/(q.date.nunique()*15),4))
q.to_csv('scripts/miner_3_20270322_efficiency_trend_signal.csv',index=False)
