import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for s in U}
p=pd.DataFrame(P).sort_index(); r=p.pct_change()
# Leave-one-out equal-weight market beta: reward assets with lower systematic equity/cross-asset exposure.
F=pd.DataFrame(index=p.index,columns=U,dtype=float)
for s in U:
 oth=r.drop(columns=s).mean(axis=1)
 z=pd.concat([r[s],oth],axis=1).dropna(); z.columns=['a','m']
 beta=z.a.rolling(60,min_periods=40).cov(z.m)/z.m.rolling(60,min_periods=40).var()
 F[s]=-beta.reindex(p.index)
for h in [1,5,10]:
 q=[]; ns=[]
 for i in range(60,len(p)-h):
  z=pd.concat([F.iloc[i],p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 q=np.array(q); print('h',h,'dates',len(q),'avg_names',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(np.mean(q),6),'ICIR',round(np.mean(q)/np.std(q,ddof=1),6),'hit',round(np.mean(q>0),4))
print('turnover',round(np.nanmean(np.abs(F.rank(pct=True).diff()).stack()),4),'valid coverage',round(F.notna().stack().mean(),4))
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 q=[]
 for i in range(60,len(p)-1):
  if lo<=p.index[i].year<=hi:
   z=pd.concat([F.iloc[i],p.iloc[i+1]/p.iloc[i]-1],axis=1).dropna()
   if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('regime',lo,hi,'n',len(q),'IC',round(np.mean(q),6) if q else None,'ICIR',round(np.mean(q)/np.std(q,ddof=1),6) if len(q)>1 else None)
print('corr proxies rev5',F.stack().corr((-r.rolling(5).sum()).stack()),'mom20',F.stack().corr(r.rolling(20).sum().stack()))
