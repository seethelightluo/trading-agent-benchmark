import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-08-26')
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.loc[:cut] for a in A}); P.index=pd.to_datetime(P.index); R=P.pct_change(); m=R.mean(axis=1); Y=pd.DataFrame({a:P[a].pct_change().shift(-1) for a in A})
def ic(f,y):
 q=[];ds=[];nn=[]
 for d in R.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(d);nn.append(len(z))
 return pd.Series(q,index=pd.to_datetime(ds)),nn
for w in [20,40,60,90]:
 cov=R.rolling(w,min_periods=max(15,w-10)).cov(m).div(m.rolling(w,min_periods=max(15,w-10)).var(),axis=0); f=-cov
 q,n=ic(f,Y);print(w,len(q),round(np.mean(n),2),round(f.stack().notna().mean(),4),round(q.mean(),6),round(q.mean()/q.std(),6),round((q>0).mean(),4),round(f.rank(pct=True,axis=1).diff().abs().mean(axis=1).mean(),4),q.groupby(q.index.year).mean().round(4).to_dict())
