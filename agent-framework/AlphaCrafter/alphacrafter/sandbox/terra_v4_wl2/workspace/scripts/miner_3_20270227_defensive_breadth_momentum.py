import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D=['XAU','US10Y','CN10Y']
def g(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None:return x
  except Exception: pass
px=pd.DataFrame({s:g(s).set_index('date')['close'] for s in U}).sort_index(); r=px.pct_change(); fr=px.shift(-1)/px-1
breadth=(r[U[:8]].lt(0).sum(axis=1)/r[U[:8]].notna().sum(axis=1)).shift(1)
mom=r.rolling(5).sum(); base=mom.sub(mom.median(axis=1),axis=0)
for th in [.375,.5,.625]:
 sig=base.where(pd.DataFrame(np.repeat((breadth.values>=th)[:,None],len(U),axis=1),index=base.index,columns=base.columns)); sig=sig[D]
 vals=[];ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=2 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(vals); print('defensive',th,'dates',len(a),'avgN',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0),'coverage',sig.notna().sum().sum()/len(U)/len(sig))
sig=base.where(pd.DataFrame(np.repeat((breadth.values>=.375)[:,None],len(U),axis=1),index=base.index,columns=base.columns)); vals=[];ns=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
a=np.array(vals);print('all .375','dates',len(a),'avgN',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0),'coverage',sig.notna().sum().sum()/len(U)/len(sig))
