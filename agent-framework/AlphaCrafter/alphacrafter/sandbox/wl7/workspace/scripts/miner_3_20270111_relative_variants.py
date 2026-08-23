import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-01-10')
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,3000)
   if x is not None and len(x):
    x=x.copy();x.date=pd.to_datetime(x.date).dt.normalize();return x.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except:pass
D={s:get(s) for s in U};D={s:x for s,x in D.items() if x is not None}
def test(n,v):
 R=pd.concat({s:d.close.pct_change(n) for s,d in D.items()},axis=1); V=pd.concat({s:d.close.pct_change().rolling(v).std() for s,d in D.items()},axis=1);F=(R.sub(R.median(axis=1),axis=0)/V).shift(1);fr=pd.concat({s:D[s].close.shift(-1)/D[s].close-1 for s in D},axis=1); a=[];ns=[]
 for dt in F.index:
  q=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:a.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'));ns.append(len(q))
 a=pd.Series(a);print(n,v,len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(252),(a>0).mean(),F.notna().mean().mean())
for n,v in [(5,20),(5,40),(10,20),(15,20),(15,40),(20,40),(20,60),(30,60)]:test(n,v)
