import numpy as np,pandas as pd,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in A:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p);d.date=pd.to_datetime(d.date);D[s]=d.set_index('date').sort_index().close
P=pd.concat(D,axis=1).sort_index().ffill();R=P.pct_change(); bench=R.mean(axis=1)
# Trend factor conditioned on breadth: in weak breadth use 5d reversal, otherwise 20d momentum.
breadth=(R>0).sum(axis=1)/len(A); weak=breadth.rolling(5).mean()<0.4
mom20=P.pct_change(20); rev5=-P.pct_change(5)
F=mom20.where(~weak,rev5).shift(1)
for h in [1,3,5,10]:
 vals=[]; ns=[]; turns=[]; prev=None
 for i in range(25,len(P)-h):
  z=pd.concat([F.iloc[i],P.pct_change(h).iloc[i+1]],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));q=F.iloc[i].rank(pct=True);turns.append(np.abs(q-(prev if prev is not None else q)).mean());prev=q
 x=np.asarray(vals);print('h',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'dailyICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4),'turn',round(np.mean(turns),4))
