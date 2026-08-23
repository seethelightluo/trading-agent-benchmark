import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
def ld(p):
 d=pd.read_csv(p);d.date=pd.to_datetime(d.date);return d.set_index('date').close.astype(float).sort_index().pct_change()
R=pd.DataFrame({s:ld(base+'/'+s+'.csv') for s in U}).join(ld('../persistent/index_data/DXY.csv').rename('D'),how='inner').loc[:'2026-07-15']
for w in [20,60,120]:
 all=[]; ranks=[]
 for t in range(w,len(R)-1):
  h=R.iloc[t-w:t];f={}
  for s in U:
   q=h[[s,'D']].dropna()
   if len(q)>=max(15,int(w*.7)): f[s]=-q[s].cov(q.D)/(q.D.var()+1e-12)
  q=pd.concat([pd.Series(f),R.iloc[t+1][U].rename('y')],axis=1).dropna()
  if len(q)>=8: all.append(spearmanr(q.iloc[:,0],q.y).statistic);ranks.append(pd.Series(f).rank(pct=True))
 a=np.array(all);turn=np.mean([(x-y).abs().mean() for x,y in zip(ranks[:-1],ranks[1:])]);print(w,'dates',len(a),'names',len(q),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'turn',turn)
