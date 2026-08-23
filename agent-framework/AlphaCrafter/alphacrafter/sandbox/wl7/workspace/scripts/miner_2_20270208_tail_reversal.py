import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-02-07')
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,3000)
   if x is not None and len(x):
    x=x.copy();x.date=pd.to_datetime(x.date).dt.normalize();return x.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except: pass
D={s:get(s) for s in U};D={s:x for s,x in D.items() if x is not None}
def calc(h):
 rows=[]
 for s,d in D.items():
  c=d.close.replace(0,np.nan);r=c.pct_change(); dn=(-r.clip(upper=0)).rolling(10,min_periods=8).std(); f=(-(c/c.shift(5)-1)/(dn*np.sqrt(10)+1e-8)).shift(1); fr=c.shift(-h)/c-1
  rows.append(pd.DataFrame({'f':f,'fr':fr,'asset':s}).reset_index())
 q=pd.concat(rows).replace([np.inf,-np.inf],np.nan).dropna();a=[]; ns=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:a.append(g.f.corr(g.fr,method='spearman'));ns.append(len(g))
 a=pd.Series(a);return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(252),(a>0).mean(),q.date.nunique()
print('assets',len(D));
for h in [1,2,5,10]:print(h,calc(h))
