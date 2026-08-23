import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy();d.date=pd.to_datetime(d.date).dt.normalize();return d.drop_duplicates('date').set_index('date').sort_index()
  except: pass
D={s:get(s) for s in U};D={s:d for s,d in D.items() if d is not None}
def fac(d,h):
 c=d.close; r=c.pct_change(); v=r.rolling(20,min_periods=15).std()
 m=(c/c.shift(20)-1)/(v*np.sqrt(20)+1e-8); long=c/c.shift(60)-1
 breadth=(r>0).astype(float).rolling(20).mean()-0.5
 f=(m*(1+0.8*np.sign(long)*breadth)).shift(1); fr=c.shift(-h)/c-1
 return pd.DataFrame({'f':f,'fr':fr}).dropna()
def calc(h,lo=None,hi=None):
 z=[]
 for s,d in D.items():
  q=fac(d,h)
  if lo:q=q[(q.index.year>=lo)&(q.index.year<=hi)]
  q['asset']=s;z.append(q.reset_index())
 q=pd.concat(z);a=[];ns=[]
 for _,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:a.append(g.f.corr(g.fr,method='spearman'));ns.append(len(g))
 a=pd.Series(a).dropna();return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(252),(a>0).mean()
print('assets',len(D),'date_range',min(d.index.min() for d in D.values()),max(d.index.max() for d in D.values()))
for h in [1,5,10,20]:print('horizon',h,'dates avg_names IC ICIR hit',calc(h))
for x in [(2020,2022),(2023,2024),(2025,2026)]:print('regime',x,calc(1,*x))
z=[]
for s,d in D.items():
 q=fac(d,1);q['asset']=s;z.append(q.reset_index())
q=pd.concat(z);piv=q.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True)
print('dates',len(piv),'coverage',piv.notna().mean().mean(),'turnover',piv.diff().abs().mean().mean())
