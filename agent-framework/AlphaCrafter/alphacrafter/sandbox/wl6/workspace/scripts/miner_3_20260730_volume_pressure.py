import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'
D={s:pd.read_csv(f'{b}/{s}.csv',parse_dates=['date']).set_index('date') for s in U}; close=pd.DataFrame({s:D[s].close for s in U}); ret=close.pct_change(); vol=pd.DataFrame({s:D[s].volume for s in U})
# volume-amplified close-location pressure, with asset-normalized log-volume surprise
clv=pd.DataFrame({s:2*(D[s].close-D[s].low)/(D[s].high-D[s].low).replace(0,np.nan)-1 for s in U})
vs=np.log1p(vol).sub(np.log1p(vol).rolling(20,min_periods=10).mean()).div(np.log1p(vol).rolling(20,min_periods=10).std())
f=-clv*vs
A=[];ns=[];ds=[];ts=[];prev=None
for dt in f.index:
 z=pd.concat([f.loc[dt],ret.shift(-1).loc[dt]],axis=1).dropna()
 if len(z)>=8:
  A.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(dt)
  q=f.loc[dt].rank(pct=True)
  if prev is not None: ts.append(np.mean(abs(q-prev).dropna()))
  prev=q
A=np.array(A);print('dates',len(A),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15,'IC',A.mean(),'ICIR',A.mean()/A.std(ddof=1),'hit',np.mean(A>0),'turnover',np.mean(ts))
for h in [5,10]:
 z=[]
 for dt in f.index:
  y=ret.shift(-1).rolling(h).sum().shift(-(h-1)).loc[dt];q=pd.concat([f.loc[dt],y],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 z=np.array(z);print(h,'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'n',len(z))
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 z=A[[lo<=x.year<=hi for x in ds]]; print(lo,hi,len(z),z.mean() if len(z) else np.nan,z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
