import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end='2027-02-24'; F={}
for s in U:
 d=get_stock_daily_data(s,days=2600)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); F[s]=d[d.date<=end].set_index('date').close
P=pd.concat(F,axis=1).sort_index(); R=P.pct_change(); bm=R.mean(axis=1); w=40
sig=pd.DataFrame(index=P.index,columns=U,dtype=float)
for i in range(w,len(R)):
 z=pd.concat([R.iloc[i-w:i],bm.iloc[i-w:i].rename('bm')],axis=1)
 for s in U:
  q=z[[s,'bm']].dropna()
  if len(q)>=20 and q.bm.var()>0:
   beta=q[s].cov(q.bm)/q.bm.var(); sig.iloc[i,sig.columns.get_loc(s)]=((q[s]-beta*q.bm)+1).prod()-1
sig=sig.shift(1); fwd=R.shift(-1); ics=[]; dates=[]; ns=[]
for dt in sig.index:
 q=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1:
  ics.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman')); dates.append(dt); ns.append(len(q))
a=pd.Series(ics,index=pd.DatetimeIndex(dates)); sd=a.std(ddof=1)
print('dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),8),'ICIR',round(a.mean()/sd*np.sqrt(252),8),'hit',round((a>0).mean(),4))
print('turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6),'coverage',round(sig.notna().sum(axis=1).mean()/15,6))
rows=sig.stack().rename('signal').reset_index(); rows.columns=['date','symbol','signal']; rows.to_csv('../persistent/factor_signals_miner_2_20270225_residual_momentum40_v2.csv',index=False)
