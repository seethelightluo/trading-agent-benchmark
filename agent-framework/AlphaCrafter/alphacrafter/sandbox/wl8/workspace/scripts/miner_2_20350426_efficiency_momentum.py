import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(f'{P}/{s}.csv',parse_dates=['date']).set_index('date')['close'].astype(float)
 px[s]=d
p=pd.DataFrame(px).sort_index(); r=np.log(p).diff()
# Efficiency-weighted momentum: directional 20d move divided by path length, then scaled by 20d return.
ret20=p.shift(1)/p.shift(21)-1
path=r.shift(1).rolling(20).apply(lambda x: np.abs(x).sum(),raw=True)
f=ret20/(path+1e-8)
# forward 10-session return after information date
fr=p.shift(-10)/p-1
ics=[]; dates=[]; ns=[]; turnovers=[]; prev=None
for dt in p.index:
 x=f.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(ic): ics.append(ic);dates.append(dt);ns.append(len(z))
  q=x.rank(pct=True)
  if prev is not None: turnovers.append(np.nanmean(np.abs(q-prev)))
  prev=q
arr=np.array(ics); print('factor=efficiency_weighted_momentum_20d');print('dates',len(arr),'avgN',np.mean(ns),'coverage',np.mean([len(f.loc[d].dropna())/15 for d in p.index]))
print('IC %.9f ICIR %.9f hit %.5f turnover %.6f'%(arr.mean(),arr.mean()/(arr.std(ddof=1)+1e-12),np.mean(arr>0),np.mean(turnovers)))
for a,b in [(0,1000),(1000,2000),(2000,3000),(3000,4100)]:
 q=arr[a:min(b,len(arr))]; print('regime',a,b,'n',len(q),'ic',q.mean() if len(q) else np.nan,'icir',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
