import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
p=pd.DataFrame(P).sort_index(); r=p.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(p.index).ffill(); vr=v.pct_change()
# resilience: subsequent returns after recent VIX shock, normalized by asset beta exposure
shock=(vr-vr.rolling(60,min_periods=40).mean())/vr.rolling(60,min_periods=40).std()
# high factor = asset return on days with VIX shock, testing rebound/resilience
f=pd.DataFrame(index=p.index,columns=U,dtype=float)
for s in U:
 z=pd.concat([r[s],shock],axis=1).dropna(); z.columns=['r','q']
 # shock-day conditional mean, exponentially weighted recent 40 observations
 f[s]=z.r.where(z.q>1).rolling(60,min_periods=15).mean().reindex(p.index)
for h in [1,5,10]:
 qs=[]; ns=[]
 for i in range(60,len(p)-h):
  z=pd.concat([f.iloc[i],p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: qs.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 q=np.array(qs); print(h,'dates',len(q),'avg_names',round(np.mean(ns),2),'IC',round(np.mean(q),6),'ICIR',round(np.mean(q)/np.std(q,ddof=1),6),'hit',round(np.mean(q>0),4))
print('turnover',round(np.nanmean(np.abs(f.rank(pct=True).diff()).stack()),4),'coverage',round(f.notna().stack().mean(),4))
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 q=[]
 for i in range(60,len(p)-1):
  if lo<=p.index[i].year<=hi:
   z=pd.concat([f.iloc[i],p.iloc[i+1]/p.iloc[i]-1],axis=1).dropna()
   if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('regime',lo,hi,'n',len(q),'ICIR',round(np.mean(q)/np.std(q,ddof=1),4) if len(q)>1 else None)
print('corr existing proxies reversal',f.stack().corr((-r.rolling(5).sum()).stack()),'momentum',f.stack().corr(r.rolling(20).sum().stack()))
