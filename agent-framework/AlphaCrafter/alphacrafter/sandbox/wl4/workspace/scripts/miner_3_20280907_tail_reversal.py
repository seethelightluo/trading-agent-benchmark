import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; cut=pd.Timestamp('2028-09-06')
P={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index();P[s]=d[d.index<=cut]
p=pd.DataFrame(P).sort_index(); r=p.pct_change(); d=r.where(r<0).rolling(20,min_periods=10).std()
f=(-p.pct_change(5)/(d+1e-8)).shift(1)
for k in [1,5,10,20]:
 q=[];ns=[]
 for t in f.index:
  z=pd.concat([f.loc[t],p.pct_change(k).shift(-k).loc[t]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 q=np.array(q); print(k,len(q),round(np.mean(ns),2),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6),round((q>0).mean(),4))
q=q[-250:];print('recent',q.mean(),q.mean()/q.std(ddof=1),'coverage',np.mean(ns)/15,'minmax',min(ns),max(ns),'turnover')
R=f.rank(axis=1,pct=True); tv=[]
for i in range(1,len(R)):
 z=pd.concat([R.iloc[i-1],R.iloc[i]],axis=1).dropna()
 if len(z)>=8:tv.append(abs(z.iloc[:,0]-z.iloc[:,1]).mean())
print(np.mean(tv),len(tv))
