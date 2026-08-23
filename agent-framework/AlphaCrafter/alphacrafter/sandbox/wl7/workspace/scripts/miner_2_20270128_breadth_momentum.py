import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-01-27')
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,3000)
   if x is not None and len(x):
    x=x.copy();x.date=pd.to_datetime(x.date).dt.normalize();return x.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
  except: pass
D={s:get(s) for s in U};D={s:x for s,x in D.items() if x is not None};C=pd.concat({s:d.close for s,d in D.items()},axis=1)
ret=C.pct_change(5); breadth=(C.pct_change(20)>0).mean(axis=1); F=ret.mul(0.5+breadth,axis=0).shift(1)

def calc(T):
 vals=[];ns=[]
 for dt in F.index:
  q=pd.concat([F.loc[dt],T.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1: vals.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'));ns.append(len(q))
 return len(vals),np.mean(ns),pd.Series(vals)
FR=pd.concat({s:D[s].close.shift(-1)/D[s].close-1 for s in D},axis=1)
n,an,arr=calc(FR); ic=arr.mean(); print('assets',len(D),'dates',n,'avg_n',round(an,2),'IC',round(ic,5),'ICIR',round(ic/arr.std(ddof=1)*np.sqrt(252),4),'hit',round((arr>0).mean(),4),'coverage',round(F.notna().mean().mean(),4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for h in [5,10,20]:
 T=pd.concat({s:D[s].close.shift(-h)/D[s].close-1 for s in D},axis=1);print('decay',h,round(calc(T)[2].mean(),5))
F.rank(axis=1,pct=True).to_csv('scripts/miner_2_20270128_breadth_signal.csv')
