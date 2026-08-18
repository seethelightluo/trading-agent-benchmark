import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2026-07-15')
def ser(path): return pd.read_csv(path,parse_dates=['date']).set_index('date').sort_index().close
p=pd.concat({s:ser('../persistent/stock_data/'+s+'.csv') for s in U},axis=1).sort_index().loc[:end]
r=p.pct_change(); d=ser('../persistent/index_data/DXY.csv').reindex(p.index).ffill(); dr=d.pct_change()
# defensive DXY beta: favor assets that historically rise when DXY falls
F=pd.DataFrame(index=p.index,columns=U,dtype=float)
for i in range(60,len(p)):
 x=dr.iloc[i-59:i+1]
 for s in U:
  y=r[s].iloc[i-59:i+1]; z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=45 and z.iloc[:,0].var()>1e-12: F.loc[p.index[i],s]=-z.iloc[:,0].cov(z.iloc[:,1])/z.iloc[:,0].var()
y=p.shift(-1)/p-1
A=[]; ns=[]; ds=[]
for dt in p.index:
 z=pd.concat([F.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q): A.append(q);ns.append(len(z));ds.append(dt)
a=np.array(A); print('dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(F.notna().mean().mean(),4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
for h in [5,10]:
 y=p.shift(-h)/p-1;a=[]
 for dt in p.index:
  z=pd.concat([F.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(a);print('h',h,'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'n',len(a))
print('turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
ds=pd.DatetimeIndex(ds)
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-15')]:
 q=a[(ds>=pd.Timestamp(lo))&(ds<=pd.Timestamp(hi))] if len(a)==len(ds) else np.array([])
 print('regime',lo,'n',len(q),'IC',round(q.mean(),6) if len(q) else None,'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)
# library proxy correlations
for n,x in {'mom20':r.rolling(20).sum(),'rev5':-r.rolling(5).sum(),'clv':(p/p.rolling(20).max()-1)}.items():print('corr',n,round(F.stack().corr(x.stack()),4))
