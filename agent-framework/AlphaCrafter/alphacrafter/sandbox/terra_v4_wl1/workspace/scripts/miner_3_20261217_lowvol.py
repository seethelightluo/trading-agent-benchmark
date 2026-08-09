import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17')
D={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv')); x.date=pd.to_datetime(x.date); x=x[x.date<=cut].sort_values('date').set_index('date'); D[s]=x.close.astype(float).pct_change()
R=pd.concat(D,axis=1).sort_index(); rows=[]
for w in [10,20,40,60]:
 F=-R.rolling(w,min_periods=max(5,w//2)).std()
 for h in [1,5,10]:
  Y=R.shift(-1).rolling(h).sum().shift(-(h-1)); vals=[]
  for dt in R.index:
   z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
   if len(z)>=8: vals.append(spearmanr(z.f,z.y).statistic)
  q=pd.Series(vals).dropna(); print('w,h',w,h,'IC %.5f ICIR %.5f hit %.3f dates %d'%(q.mean(),q.mean()/q.std(ddof=1), (q>0).mean(),len(q)))
print('cut',R.index.max(),'instruments',len(U))
