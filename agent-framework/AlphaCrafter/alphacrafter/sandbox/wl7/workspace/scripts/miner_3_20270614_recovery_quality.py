import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-06-13')
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
D={s:load(s) for s in U}; D={s:d for s,d in D.items() if d is not None}; rows=[]
# Recovery-quality: medium-horizon return normalized by worst trailing drawdown.
# Positive trend with shallow drawdowns is favored; shifted one completed session.
for s,d in D.items():
 c=d.close.astype(float); r=c.pct_change(); ret=c/c.shift(20)-1
 peak=c.rolling(20,min_periods=15).max(); dd=c/peak-1
 mdd=(-dd).rolling(20,min_periods=15).max()
 vol=r.rolling(20,min_periods=15).std()
 f=(ret/(mdd+0.01))/(vol+1e-12)
 rows.append(pd.DataFrame({'date':c.index,'asset':s,'f':f.shift(1),'fr1':c.shift(-1)/c-1,'fr5':c.shift(-5)/c-1,'fr10':c.shift(-10)/c-1}))
q=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna(subset=['f','fr1'])
def stats(x,col):
 z=[]; ns=[]
 for _,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g[col].nunique()>1:
   z.append(g.f.corr(g[col],method='spearman')); ns.append(len(g))
 z=pd.Series(z).dropna(); return len(z),float(np.mean(ns)),float(z.mean()),float(z.mean()/z.std(ddof=1)*np.sqrt(252)) if len(z)>1 else np.nan,float((z>0).mean())
print('assets',len(D),'dates',q.date.nunique(),'avg_n',q.groupby('date').size().mean(),'coverage',len(q)/(q.date.nunique()*15))
for col in ['fr1','fr5','fr10']: print(col,stats(q,col))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',a,b,stats(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)],'fr1'))
p=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True); print('turnover',float(p.diff().abs().mean().mean()))
q.to_csv('scripts/miner_3_20270614_recovery_quality_signal.csv',index=False)
