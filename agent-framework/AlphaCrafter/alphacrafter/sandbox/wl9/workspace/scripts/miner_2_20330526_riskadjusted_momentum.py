import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  d=d[['date','close']].copy(); d.date=pd.to_datetime(d.date); d=d.drop_duplicates('date').set_index('date').sort_index()
  r=d.close.pct_change(20); v=d.close.pct_change().rolling(20).std()*np.sqrt(252)
  P[s]=pd.DataFrame({'fac':(r/(v+.05)).shift(1),'r10':d.close.shift(-10)/d.close-1,'r20':d.close.shift(-20)/d.close-1,'r40':d.close.shift(-40)/d.close-1,'r60':d.close.shift(-60)/d.close-1})
X=pd.concat(P,axis=1); print('symbols',len(P),'dates',X.index.min(),X.index.max())
for h in [10,20,40,60]:
 A=X.xs('fac',level=1,axis=1); B=X.xs('r'+str(h),level=1,axis=1); cs=[]; dates=[]; ns=[]
 for dt in X.index:
  z=pd.concat([A.loc[dt],B.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   cs.append(z.iloc[:,0].corr(z.iloc[:,1])); dates.append(dt); ns.append(len(z))
 a=np.array(cs); print('H',h,'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
 if h==60:
  for lab,lo,hi in [('2024','2024-01-01','2024-12-31'),('2027-29','2027-01-01','2029-12-31'),('2030','2030-01-01','2030-12-31'),('2031-32','2031-01-01','2032-12-31'),('2033','2033-01-01','2033-05-25')]:
   q=a[(np.array(dates)>=pd.Timestamp(lo))&(np.array(dates)<=pd.Timestamp(hi))]; print('REG',lab,'n',len(q),'IC',q.mean() if len(q) else np.nan)
# cross-sectional sign turnover, broad coverage
valid=A.notna().sum(axis=1); print('coverage',valid.mean()/len(U),'valid dates',int((valid>=8).sum()))
print('turnover',((A>0).astype(float).diff().abs().sum(axis=1)/(2*valid)).dropna().mean())
