import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17'); D={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv')); x.date=pd.to_datetime(x.date); x=x[x.date<=cut].sort_values('date').set_index('date'); D[s]=x.close.astype(float).pct_change()
R=pd.concat(D,axis=1).sort_index(); b=R.mean(axis=1); out=[]
for w in [20,60,120]:
 cov=R.rolling(w,min_periods=max(10,w//2)).cov(b); var=b.rolling(w,min_periods=max(10,w//2)).var(); beta=cov.div(var,axis=0)
 # low beta as defensive; also beta change
 for name,F in [('neg_beta',-beta),('beta_change',-beta.diff(20))]:
  Y=R.shift(-1)
  vals=[]; ns=[]
  for dt in R.index:
   z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
   if len(z)>=8: vals.append(spearmanr(z.f,z.y).statistic); ns.append(len(z))
  q=pd.Series(vals).dropna(); print('w',w,name,'IC %.5f ICIR %.5f hit %.3f dates %d avg_names %.2f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),len(q),np.mean(ns)))
print('range',R.index.min(),R.index.max(),'assets',len(U))
