import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); return d.set_index('date').close
px=pd.concat({s:load(s) for s in U},axis=1).sort_index(); r=px.pct_change()
# low idiosyncratic volatility: negative residual standard deviation to global equal-weight market
m=r.mean(axis=1); resid=r.sub(m,axis=0)
fac=-resid.rolling(20,min_periods=15).std()
for h in [1,5,10]:
 a=[]; ns=[]; ds=[]
 for i in range(len(px)-h):
  q=pd.concat([fac.iloc[i],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
  if len(q)>=8:a.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q));ds.append(px.index[i])
 a=np.array(a); print('low_idio_vol',h,'dates',len(a),'avgN',np.mean(ns),'coverage',np.mean(ns)/15,'IC',a.mean(),'ICIR',a.mean()/a.std(),'hit',np.mean(a>0))
 for y in [2020,2021,2022,2023,2024,2025,2026]:
  b=a[[d.year==y for d in ds]]
  if len(b):print(y,len(b),round(b.mean(),5),round(b.mean()/b.std(),5))
print('turnover',fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
