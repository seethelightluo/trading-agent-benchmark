import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>150:
  d=d[['date','close']].copy(); d.date=pd.to_datetime(d.date); d=d.drop_duplicates('date').set_index('date').sort_index()
  ret=d.close.pct_change(); r60=d.close.pct_change(60)
  downside=np.minimum(ret,0.0)
  down=np.sqrt((downside**2).rolling(60).mean())*np.sqrt(252)
  fac=(-r60/(down+.05)).shift(1)
  P[s]=pd.DataFrame({'fac':fac,'r10':d.close.shift(-10)/d.close-1,'r20':d.close.shift(-20)/d.close-1,'r40':d.close.shift(-40)/d.close-1,'r60':d.close.shift(-60)/d.close-1})
X=pd.concat(P,axis=1); print('symbols',len(P),'dates',X.index.min(),X.index.max())
A=X.xs('fac',level=1,axis=1)
for h in [10,20,40,60]:
 B=X.xs('r'+str(h),level=1,axis=1); cs=[]; dates=[]; ns=[]
 for dt in X.index:
  z=pd.concat([A.loc[dt],B.loc[dt]],axis=1).dropna()
  if len(z)>=8: cs.append(z.iloc[:,0].corr(z.iloc[:,1])); dates.append(dt); ns.append(len(z))
 a=np.array(cs); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 if h==60:
  for lab,lo,hi in [('2024','2024-01-01','2024-12-31'),('2027-29','2027-01-01','2029-12-31'),('2030','2030-01-01','2030-12-31'),('2031-32','2031-01-01','2032-12-31'),('2033','2033-01-01','2033-06-08')]:
   q=a[(pd.to_datetime(dates)>=pd.Timestamp(lo))&(pd.to_datetime(dates)<=pd.Timestamp(hi))]; print('REG',lab,'n',len(q),'IC',round(q.mean(),6) if len(q) else np.nan)
valid=A.notna().sum(axis=1); print('coverage',round(valid.mean()/len(U),6),'valid dates',int((valid>=8).sum()))
print('turnover',round(((A>0).astype(float).diff().abs().sum(axis=1)/(2*valid)).dropna().mean(),6))
out=A.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20330609_downside_asymmetry_reversal_signal.csv',index=False)
