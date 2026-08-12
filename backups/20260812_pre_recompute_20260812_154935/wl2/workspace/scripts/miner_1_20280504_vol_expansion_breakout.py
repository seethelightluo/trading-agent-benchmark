import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for a in assets:
 f='../persistent/stock_data/'+a+'.csv'
 d=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index()
 P[a]=d['close'].replace(0,np.nan)
px=pd.DataFrame(P).sort_index(); r=np.log(px).diff()
# Volatility-expansion breakout: medium trend, amplified by recent/long volatility expansion, fully lagged
rv20=r.rolling(20,min_periods=15).std(); rv60=r.rolling(60,min_periods=40).std()
f=(np.log(px/px.shift(20))*(rv20/rv60)).shift(1)
f=f.replace([np.inf,-np.inf],np.nan)
ics=[]; turnovers=[]; ns=[]
for i in range(len(px)-1):
 x=f.iloc[i]; y=r.iloc[i+1]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
  if i>0:
   q=f.iloc[i-1].rank(pct=True); q2=x.rank(pct=True)
   zz=pd.concat([q,q2],axis=1).dropna()
   turnovers.append(np.abs(zz.iloc[:,0]-zz.iloc[:,1]).mean())
a=np.array(ics); print('idea=volatility_expansion_breakout');print('dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR_mean_std',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'turnover',np.mean(turnovers),'coverage',np.mean(ns)/15,'period',px.index.min().date(),px.index[-1].date())
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2025-12-31'),('2026-01-01','2027-12-31'),('2028-01-01','2028-05-03')]:
 mask=(px.index[1:-1]>=lo)&(px.index[1:-1]<=hi); b=a[mask[:len(a)]]
 if len(b): print(lo,hi,len(b),b.mean(),b.mean()/b.std(ddof=1) if len(b)>1 else np.nan)
