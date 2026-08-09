import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end='2027-02-24'; F={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); F[s]=d[d.date<=end].set_index('date').close
P=pd.concat(F,axis=1).sort_index(); R=P.pct_change(); bm=R.mean(axis=1); w=20
sig=pd.DataFrame(index=P.index,columns=U,dtype=float)
for i in range(w,len(R)):
 for s in U:
  q=pd.concat([R[s].iloc[i-w:i],bm.iloc[i-w:i]],axis=1).dropna(); q.columns=['x','b']
  if len(q)>=20 and q.b.var()>0:
   beta=q.x.cov(q.b)/q.b.var(); sig.loc[sig.index[i],s]=((q.x-beta*q.b)+1).prod()-1
sig=sig.shift(1)
for h in [1,5,10,20]:
 fw=P.pct_change(h).shift(-h); ics=[];ns=[]
 for dt in sig.index:
  q=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1: ics.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'));ns.append(len(q))
 a=pd.Series(ics); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1),8),'hit',round((a>0).mean(),4))
print('turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6),'coverage',round(sig.notna().sum(axis=1).mean()/15,6))
sig.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('../persistent/factor_signals_miner_2_20270225_residual_momentum20.csv',index=False)
