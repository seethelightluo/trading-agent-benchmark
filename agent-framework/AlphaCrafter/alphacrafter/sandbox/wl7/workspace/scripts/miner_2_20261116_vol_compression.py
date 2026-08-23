import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index()
  except: pass
D={s:fetch(s) for s in U}; D={s:d for s,d in D.items() if d is not None}
def feat(d,h):
 c=d.close.replace(0,np.nan); r=c.pct_change(); v=r.rolling(20,min_periods=15).std(); long=r.rolling(60,min_periods=45).std(); f=(-(v/(long+1e-9))).shift(1); fr=c.shift(-h)/c-1
 return pd.DataFrame({'f':f,'fr':fr}).replace([np.inf,-np.inf],np.nan).dropna()
def calc(h,lo='2020-01-01',hi='2026-11-15'):
 z=[]
 for s,d in D.items():
  q=feat(d,h);q=q.loc[(q.index>=lo)&(q.index<=hi)];q['asset']=s;z.append(q.reset_index())
 q=pd.concat(z); a=[]; ns=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1:a.append(g.f.corr(g.fr,method='spearman'));ns.append(len(g))
 a=pd.Series(a);return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(252),(a>0).mean()
print('assets',len(D))
for h in [1,5,10,20]:print('horizon',h,calc(h))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-11-15')]:print('regime',lo[:4]+'-'+hi[:4],calc(1,lo,hi))
