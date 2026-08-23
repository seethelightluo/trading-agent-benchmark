import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}; P=pd.DataFrame(P).sort_index().loc[:'2035-05-23']
r=P.pct_change(20); ds=r.std(axis=1); mn=ds.rolling(60,min_periods=30).min(); mx=ds.rolling(60,min_periods=30).max(); d=((ds-mn)/(mx-mn)).replace([np.inf,-np.inf],np.nan).rolling(5,min_periods=3).mean()
lo=P.rolling(240,min_periods=168).min(); hi=P.rolling(240,min_periods=168).max(); F=(-(P-lo)/(hi-lo)).mul(.5+d,axis=0).shift(1)
rows=[]
for dt in F.index:
 y=P.shift(-10).loc[dt]/P.loc[dt]-1; q=pd.concat([F.loc[dt],y],axis=1).dropna()
 if len(q)>=8: rows.append((dt,len(q),spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
a=pd.DataFrame(rows,columns=['date','n','ic']); print('dates',len(a),'avgN',a.n.mean(),'coverage',a.n.sum()/(len(a)*15)); print('full',a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1),(a.ic>0).mean())
for k in [120,260,520]:
 q=a.tail(k); print('recent',k,q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean())
