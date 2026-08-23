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
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}; rows=[]
for s,d in D.items():
 c=d.close.astype(float); r=c.pct_change(); v=r.rolling(20,min_periods=15).std()
 # volatility-normalized short-term reversal, lagged one session
 f=-r/(v+1e-12)
 rows.append(pd.DataFrame({'date':c.index,'asset':s,'f':f.shift(1),'close':c.values}))
q=pd.concat(rows,ignore_index=True); q['fr']=q.groupby('asset').close.shift(-1)/q.close-1
q=q.replace([np.inf,-np.inf],np.nan).dropna(subset=['f','fr'])
def stats(x):
 z=[]; ns=[]
 for dt,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:z.append(g.f.corr(g.fr,method='spearman'));ns.append(len(g))
 z=pd.Series(z);return len(z),round(np.mean(ns),2),round(z.mean(),5),round(z.mean()/z.std(ddof=1)*np.sqrt(252),4),round((z>0).mean(),4)
print('assets',len(D),'dates',q.date.nunique(),'coverage',len(q)/(q.date.nunique()*len(D)))
for h in [1,5,10,20]:
 x=q[['date','asset','f','close']].copy();x['fr']=q.groupby('asset').close.shift(-h)/q.close-1;print('horizon',h,stats(x.dropna()))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]:print('regime',a,b,stats(q[(q.date.dt.year>=a)&(q.date.dt.year<=b)]))
r=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True);print('turnover',r.diff().abs().mean().mean());q[['date','asset','f']].to_csv('scripts/miner_1_20270224_reversal_signal.csv',index=False)
