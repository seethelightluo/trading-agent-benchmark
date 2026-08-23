import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'
r=pd.DataFrame({s:pd.read_csv(f'{b}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).pct_change(); d=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close'].pct_change(); v=d.rolling(60,min_periods=45).var()
beta=pd.DataFrame({s:-r[s].rolling(60,min_periods=45).cov(d)/v for s in U}); f=beta.mul(d,axis=0) # dollar shock x hedge beta
ys=r.shift(-1); A=[]; ns=[]; ds=[]; prev=None; ts=[]
for dt in beta.index:
 z=pd.concat([f.loc[dt],ys.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  A.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(dt)
  q=f.loc[dt].rank(pct=True)
  if prev is not None:ts.append(np.mean(abs(q-prev).dropna()))
  prev=q
A=np.array(A);print('dates',len(A),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15,'IC',A.mean(),'ICIR',A.mean()/A.std(ddof=1),'hit',np.mean(A>0),'turnover',np.mean(ts))
for h in [5,10]:
 z=[]
 for dt in beta.index:
  y=r.shift(-1).rolling(h).sum().shift(-(h-1)).loc[dt];q=pd.concat([f.loc[dt],y],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 z=np.array(z);print(h,'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'n',len(z))
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 z=A[[lo<=x.year<=hi for x in ds]];print(lo,hi,len(z),z.mean(),z.mean()/z.std(ddof=1))
