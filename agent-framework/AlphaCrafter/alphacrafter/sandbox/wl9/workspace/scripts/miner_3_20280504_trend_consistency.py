import numpy as np, pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-05-03'); D={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index().close.astype(float); D[s]=x.loc[:end]
px=pd.DataFrame(D).sort_index(); R=px.pct_change(); out=[]; W=20
for i in range(W+1,len(R)-10):
 hist=R.iloc[i-W:i]
 # trend consistency: trailing return times fraction of positive days, centered around 0.5
 ret=px.iloc[i-1].div(px.iloc[i-W-1])-1
 consistency=(hist>0).mean(axis=0)
 f=ret*(2*consistency-1)
 for h in [1,5,10]:
  y=px.iloc[i+h].div(px.iloc[i])-1; q=pd.concat([f,y.rename('y')],axis=1).dropna()
  if len(q)>=8: out.append((R.index[i],h,len(q),spearmanr(q.iloc[:,0],q.y).statistic))
A=pd.DataFrame(out,columns=['date','h','n','ic']); print('period',px.index.min().date(),end.date(),'dates',len(R),'instruments',len(D),'observations',len(A),'coverage',round(A.n.mean()/15,4))
for h,g in A.groupby('h'):
 print('horizon',h,'dates',len(g),'mean_n',round(g.n.mean(),2),'IC',round(g.ic.mean(),6),'ICIR',round(g.ic.mean()/g.ic.std(ddof=1),6),'hit',round((g.ic>0).mean(),4))
 for name,cut in [('online','2026-07-16'),('recent','2027-05-04')]:
  q=g[g.date>=pd.Timestamp(cut)]; print(name,'dates',len(q),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))
