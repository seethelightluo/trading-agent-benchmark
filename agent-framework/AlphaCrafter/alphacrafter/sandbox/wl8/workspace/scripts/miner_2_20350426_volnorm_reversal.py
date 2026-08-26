import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P='../persistent/stock_data'
p=pd.DataFrame({s:pd.read_csv(f'{P}/{s}.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index(); r=np.log(p).diff()
# lagged short-horizon reversal normalized by recent risk, interpretable and complementary to trend
f=-r.shift(1).rolling(5).sum()/(r.shift(1).rolling(20).std()+1e-8)
fr=p.shift(-10)/p-1; vals=[]; ns=[]; ts=[]; prev=None
for dt in p.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  v=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(v): vals.append(v);ns.append(len(z));ts.append(dt)
arr=np.array(vals); print('factor=volnorm_reversal_5d');print('dates',len(arr),'avgN',np.mean(ns),'coverage',np.mean([len(f.loc[d].dropna())/15 for d in p.index])); print('IC %.9f ICIR %.9f hit %.5f'%(arr.mean(),arr.mean()/arr.std(ddof=1),np.mean(arr>0)))
for a,b in [(0,1000),(1000,2000),(2000,3000),(3000,4100)]:
 q=arr[a:min(b,len(arr))];print('regime',a,b,'n',len(q),'ic',q.mean() if len(q) else np.nan,'icir',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
# rank turnover
for dt in ts:
 pass
