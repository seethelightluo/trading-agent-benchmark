import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index().ffill(); r=np.log(P).diff()
vol=r.rolling(40,min_periods=25).std(); mom=np.log(P/P.shift(20))
short=r.rolling(10,min_periods=8).std(); long=r.rolling(60,min_periods=40).std()
sig=(mom/vol).where(short.shift(1)<long.shift(1)).shift(1)
for h in [1,5,10,20]:
 Y=np.log(P.shift(-h)/P); vals=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],Y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c): vals.append(c); ns.append(len(z))
 a=pd.Series(vals); n=len(a); ic=a.mean(); ir=ic/a.std(ddof=1) if n>1 else np.nan
 print('H',h,'dates',n,'avgN',np.mean(ns) if ns else 0,'IC',ic,'ICIR',ir,'hit',(a>0).mean() if n else 0,'thirds',[a.iloc[i:j].mean() for i,j in [(0,n//3),(n//3,2*n//3),(2*n//3,n)]])
print('assets',len(P.columns),'period',P.index.min(),P.index.max(),'coverage',sig.notna().sum(axis=1).mean()/15,'turnover',sig.rank(axis=1,pct=True).diff().abs().sum(axis=1).mean()/2)
