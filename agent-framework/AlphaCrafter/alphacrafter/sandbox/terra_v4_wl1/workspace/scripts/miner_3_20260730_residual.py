import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index(); R=P.pct_change(); bench=R['SPX']
# 60d residual trend: cumulative asset return minus rolling beta times SPX cumulative return, beta estimated past 60 daily returns
cov=R.rolling(60,min_periods=45).cov(bench); var=bench.rolling(60,min_periods=45).var(); beta=cov.div(var,axis=0)
f=(P/P.shift(20)-1)-beta*(P['SPX']/P['SPX'].shift(20)-1).values[:,None]
rows=[]
for dt in f.index:
 for h in [1,5,10]:
  z=pd.concat([f.loc[dt],(P.shift(-h).loc[dt]/P.loc[dt]-1)],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
d=pd.DataFrame(rows,columns=['date','h','n','ic']); print('dates',d[d.h==1].date.nunique(),'avgN',d[d.h==1].n.mean(),'coverage',d[d.h==1].n.mean()/15)
for h in [1,5,10]:
 q=d[d.h==h].ic; print(h,len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252),(q>0).mean())
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 q=d[(d.h==1)&(d.date>=a)&(d.date<=b)].ic; print(a,len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252))
print('turn',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
