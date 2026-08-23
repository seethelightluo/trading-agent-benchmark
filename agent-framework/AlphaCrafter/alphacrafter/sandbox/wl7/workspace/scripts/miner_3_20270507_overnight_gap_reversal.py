import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-05-06')
def f(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d): d=d.copy();d.date=pd.to_datetime(d.date).dt.normalize();return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except: pass
D={s:f(s) for s in U};D={s:d for s,d in D.items() if d is not None};rows=[]
for s,d in D.items():
 o,c=d.open.astype(float),d.close.astype(float); gap=o/c.shift(1)-1; vol=c.pct_change().rolling(20,min_periods=15).std(); sig=-(gap.shift(1))/(vol.shift(1)+1e-12)
 rows.append(pd.DataFrame({'date':c.index,'asset':s,'f':sig.values,'fr':(c.shift(-1)/c-1).values}))
q=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
def st(x):
 z=[];ns=[]
 for _,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1:z.append(g.f.corr(g.fr,method='spearman'));ns.append(len(g))
 z=pd.Series(z);return len(z),np.mean(ns),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252),(z>0).mean()
print('assets',len(D),'dates',q.date.nunique(),'coverage',len(q)/(q.date.nunique()*15),'daily',st(q))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]:print('regime',a,b,st(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)]))
print('turnover',q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True).diff().abs().mean().mean())
q.to_csv('scripts/miner_3_20270507_overnight_gap_signal.csv',index=False)
