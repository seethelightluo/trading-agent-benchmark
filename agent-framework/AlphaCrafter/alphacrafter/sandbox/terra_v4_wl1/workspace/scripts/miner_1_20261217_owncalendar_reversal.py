import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17'); D={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv')); x.date=pd.to_datetime(x.date); x=x[x.date<=cut].sort_values('date').set_index('date'); D[s]=x.close.astype(float).pct_change()
R=pd.concat(D,axis=1,sort=True).sort_index()
def ownroll(fun,w): return R.apply(lambda s: fun(s.dropna(),w).reindex(R.index))
def rolling_sum(s,w): return s.rolling(w,min_periods=w).sum()
def rolling_std(s,w): return s.rolling(w,min_periods=w).std()
def ev(F,h):
 Y=R.shift(-1).rolling(h,min_periods=h).sum().shift(-(h-1)); q=[];ns=[];ds=[]
 for dt in R.index:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(dt)
 q=pd.Series(q,index=ds);return q.mean(),q.mean()/q.std(ddof=1),len(q),np.mean(ns),(q>0).mean()
for w in [3,5,10,20,40,60]:
 s=ownroll(rolling_sum,w);v=ownroll(rolling_std,w)
 for name,F in [('reversal',-s),('vol_reversal',-s/v)]:
  print(name,w,'1d',*[round(x,6) if isinstance(x,float) else x for x in ev(F,1)],'5d',*[round(x,6) if isinstance(x,float) else x for x in ev(F,5)])
