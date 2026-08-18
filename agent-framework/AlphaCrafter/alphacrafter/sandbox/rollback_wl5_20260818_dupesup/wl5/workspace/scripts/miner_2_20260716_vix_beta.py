import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index(); ar=p.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(p.index).ffill(); vr=v.pct_change()
# continuous VIX beta: negative sensitivity, rolling correlation times volatility ratio
fac=pd.DataFrame(index=p.index)
for s in U: fac[s]=-(ar[s].rolling(40,min_periods=25).corr(vr)*ar[s].rolling(40,min_periods=25).std()/vr.rolling(40,min_periods=25).std())
for h in [1,5,10]:
 x=[];ns=[];cs=[]
 for i in range(40,len(p)-h):
  z=pd.concat([fac.iloc[i],p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));cs.append(len(z)/15)
 q=np.array(x);print(h,'N',len(q),'mean names',np.mean(ns),'coverage',np.mean(cs),'IC',np.mean(q),'ICIR',np.mean(q)/np.std(q,ddof=1),'hit',np.mean(q>0))
print('turnover',np.nanmean(np.abs(fac.rank(pct=True).diff()).stack()))
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 q=[]
 for i in range(40,len(p)-1):
  if lo<=p.index[i].year<=hi:
   z=pd.concat([fac.iloc[i],p.iloc[i+1]/p.iloc[i]-1],axis=1).dropna()
   if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(lo,hi,len(q),np.mean(q) if q else np.nan)
print('corr momentum',fac.stack().corr(ar.rolling(20).sum().stack()))
