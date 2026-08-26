import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-10-06')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except Exception: pass
 return None
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
rows=[]
for s,d in D.items():
 c=d.close.astype(float); r=c.pct_change(); vol=r.rolling(30,min_periods=20).std().shift(1)
 # candidate: trend strength penalized by recent downside asymmetry; all information lagged one session
 mom=c.pct_change(30).shift(1)
 down=(r.clip(upper=0)**2).rolling(30,min_periods=20).mean().shift(1)**0.5
 up=(r.clip(lower=0)**2).rolling(30,min_periods=20).mean().shift(1)**0.5
 f=(mom/(vol+1e-12))*(up/(down+up+1e-12))
 rows.append(pd.DataFrame({'date':c.index,'asset':s,'f':f}))
q=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
def stat(x):
 z=[]; ns=[]
 for dt,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:
   z.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 z=pd.Series(z)
 return len(z),np.mean(ns),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252),(z>0).mean()
for h in [1,5,10,20]:
 xx=[]
 for s,d in D.items(): xx.append(pd.DataFrame({'date':d.index,'asset':s,'f':q[q.asset==s].set_index('date').f.reindex(d.index),'fr':d.close.shift(-h)/d.close-1}))
 print('horizon',h,stat(pd.concat(xx,ignore_index=True).dropna()))
xx=[]
for s,d in D.items(): xx.append(pd.DataFrame({'date':d.index,'asset':s,'f':q[q.asset==s].set_index('date').f.reindex(d.index),'fr':d.close.shift(-10)/d.close-1}))
xx=pd.concat(xx,ignore_index=True).dropna()
for a,b in [(2020,2022),(2023,2024),(2025,2027)]: print('regime',a,b,stat(xx[(xx.date.dt.year>=a)&(xx.date.dt.year<=b)]))
print('recent_63',stat(xx[xx.date>=CUT-pd.Timedelta(days=100)]))
print('assets',len(D),'valid_dates',xx.date.nunique(),'avg_n',xx.groupby('date').size().mean(),'coverage',len(xx)/(xx.date.nunique()*15))
print('turnover',q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True).diff().abs().mean().mean())
q.to_csv('scripts/miner_1_20271007_asymmetry_trend_signal.csv',index=False)
