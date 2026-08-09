import numpy as np,pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
px=pd.DataFrame(P); r=px.pct_change()
fac=-(r.where(r<0,0).pow(2).rolling(30,min_periods=20).mean().pow(.5)/r.rolling(30,min_periods=20).std())
for h in [1,5,10]:
 fwd=px.shift(-h)/px-1; out=[]
 for d in fac.index:
  z=pd.concat([fac.loc[d],fwd.loc[d]],axis=1).dropna()
  if len(z)>=8: out.append((d,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
 q=pd.DataFrame(out,columns=['date','n','ic']); ic=q.ic.mean(); print('H',h,'dates',len(q),'avgN',q.n.mean(),'coverage',q.n.mean()/15,'IC',ic,'ICIR',ic/q.ic.std(),'hit',(q.ic>0).mean())
rank=fac.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean(axis=1).dropna().mean(),'period',fac.index.min(),fac.index.max())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-15'),('2026-07-16','2027-02-24')]:
 fwd=px.shift(-1)/px-1; out=[]
 for d in fac.loc[lo:hi].index:
  z=pd.concat([fac.loc[d],fwd.loc[d]],axis=1).dropna()
  if len(z)>=8:out.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print(lo,hi,len(out),np.nanmean(out))
