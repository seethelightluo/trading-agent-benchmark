import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A},axis=1).sort_index()
r=np.log(px).diff(); eq=r[['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX']].mean(axis=1)
# residual medium momentum after removing contemporaneous equity beta using rolling covariance; lag signal
cov=r.rolling(60,min_periods=40).cov(eq); var=eq.rolling(60,min_periods=40).var(); beta=cov.div(var,axis=0)
res=r.sub(beta.mul(eq,axis=0)); f=res.rolling(20,min_periods=15).sum().shift(1); y=r.shift(-1)
ics=[];ns=[];dates=[];tos=[];prev=None
for dt in px.index:
 if dt>pd.Timestamp('2028-05-03'):break
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));dates.append(dt);q=f.loc[dt].rank(pct=True)
  if prev is not None: tos.append(np.abs(q-prev).dropna().mean())
  prev=q
x=np.array(ics); print('idea=equity_beta_residual_momentum');print('dates',len(x),'avgN',np.mean(ns),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',np.mean(x>0),'turnover',np.mean(tos),'coverage',np.mean(ns)/15)
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2025-12-31'),('2026-01-01','2027-12-31'),('2028-01-01','2028-05-03')]:
 b=x[(np.array(dates)>=pd.Timestamp(lo))&(np.array(dates)<=pd.Timestamp(hi))]; print(lo,hi,len(b),b.mean(),b.mean()/b.std(ddof=1))
