import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17'); D={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv')); x.date=pd.to_datetime(x.date); x=x[x.date<=cut].sort_values('date').set_index('date'); D[s]=x.close.astype(float).pct_change()
R=pd.concat(D,axis=1,sort=True).sort_index();
def evalf(F,h):
 Y=sum(R.shift(-k) for k in range(1,h+1)); qs=[]; ns=[]; ds=[]
 for dt in R.index:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8: qs.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(dt)
 q=pd.Series(qs,index=ds); return q.mean(),q.mean()/q.std(ddof=1),len(q),np.mean(ns),(q>0).mean()
for w in [3,5,10,20,40,60]:
 candidates={'rev':-R.rolling(w,min_periods=w).sum(),'mom':R.rolling(w,min_periods=w).sum(),'volrev':-R.rolling(w,min_periods=w).sum()/R.rolling(w,min_periods=w).std(),'sharpmom':R.rolling(w,min_periods=w).mean()/R.rolling(w,min_periods=w).std()}
 for n,F in candidates.items():
  a=evalf(F,1); print(n,w,*(round(x,5) if isinstance(x,float) else x for x in a))
