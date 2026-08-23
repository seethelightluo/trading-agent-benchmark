import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-02-21')
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,3000)
   if x is not None and len(x):
    x=x.copy();x.date=pd.to_datetime(x.date).dt.normalize();return x.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except: pass
D={s:fetch(s) for s in U};D={s:x for s,x in D.items() if x is not None}
def make(h):
 a=[]
 for s,d in D.items():
  c=d.close;r=c.pct_change(); f=((c.pct_change(20)*(0.5+r.gt(0).rolling(20,min_periods=12).mean()))/(r.rolling(20,min_periods=12).std()*np.sqrt(20)+1e-8)).shift(1); fr=c.shift(-h)/c-1
  a.append(pd.DataFrame({'f':f,'fr':fr,'asset':s}))
 return pd.concat(a).replace([np.inf,-np.inf],np.nan).dropna().reset_index(names='date')
def calc(q):
 v=[];n=[]
 for _,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:v.append(g.f.corr(g.fr,method='spearman'));n.append(len(g))
 x=pd.Series(v);return len(x),np.mean(n),x.mean(),x.mean()/x.std(ddof=1)*np.sqrt(252),(x>0).mean()
print('assets',len(D),'end',max(x.index.max() for x in D.values()))
for h in [1,5,10,20]:print(h,calc(make(h)))
q=make(1)
for lo,hi in [(2020,2022),(2023,2024),(2025,2027)]:print(lo,hi,calc(q[(q.date.dt.year>=lo)&(q.date.dt.year<=hi)]))
print('coverage',len(q)/sum(len(x) for x in D.values()),'dates',q.date.nunique())
