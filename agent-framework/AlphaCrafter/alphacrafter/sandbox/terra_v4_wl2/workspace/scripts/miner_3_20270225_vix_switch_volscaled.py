import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   x=f(s,days=5000)
   if x is not None:return x
  except Exception: pass
px=pd.DataFrame({s:get(s).set_index('date')['close'] for s in U}).sort_index(); r=px.pct_change(); v=get('VIX').set_index('date')['close'].reindex(px.index).ffill(); shock=(v.pct_change(5)>0.10).shift(1).fillna(False).astype(bool)
normal=r.rolling(10).sum(); rev=-r.rolling(3).sum(); cond=pd.DataFrame({s:shock for s in U},index=px.index,dtype=bool)
sig=normal.where(~cond,rev).shift(1); sig=sig/r.rolling(20).std().replace(0,np.nan); fr=px.shift(-1)/px-1
for h in [1,5,10]:
 fwd=px.shift(-h)/px-1; vals=[]; ns=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(vals); print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1),6),'hit',round(np.mean(a>0),4),'coverage',round(sig.notna().sum().sum()/(len(U)*len(sig)),4))
zall=[]
for d in sig.index:
 z=pd.concat([sig.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:zall.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
q=pd.DataFrame(zall,columns=['date','ic']);print(q.groupby(q.date.dt.year).ic.agg(['mean','count']).to_string());print('shock share',round(shock.mean(),4))
