import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
P=pd.concat([pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.rename(s) for s in U],axis=1,sort=True).loc[:end]
R=P.pct_change(fill_method=None); M=R.median(axis=1)
# rolling beta computed with aligned per-series pairwise moments, then residual 5d cumulative return
F=pd.DataFrame(index=P.index,columns=U,dtype=float)
for s in U:
 x=R[s]; cov=x.rolling(60,min_periods=30).cov(M); vv=M.rolling(60,min_periods=30).var(); b=cov/(vv+1e-12)
 F[s]=(x-b*M).rolling(5,min_periods=5).sum()
Y={h:P.pct_change(h,fill_method=None).shift(-h) for h in [1,5,10]}
for h,y in Y.items():
 q=[]; ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 q=np.array(q);print('h',h,'dates',len(q),'avgN',np.mean(ns),'coverage',sum(ns)/(len(F)*15),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
q=[]
for d in F.index:
 z=pd.concat([F.loc[d],Y[1].loc[d]],axis=1).dropna()
 if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
q=np.array(q)
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 w=[]
 for d in F.loc[a:b].index:
  z=pd.concat([F.loc[d],Y[1].loc[d]],axis=1).dropna()
  if len(z)>=8:w.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 w=np.array(w);print('regime',a,b,len(w),w.mean(),w.mean()/w.std(ddof=1))
print('turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
