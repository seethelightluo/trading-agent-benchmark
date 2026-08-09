import pandas as pd,numpy as np
from scipy.stats import spearmanr
W=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in W}
px=pd.DataFrame(d).sort_index().loc[:'2033-04-27']; r=px.pct_change()
# Intermediate-horizon acceleration: recent 20d return minus preceding 40d return,
# volatility scaled. Lagged one session to avoid look-ahead.
recent=r.rolling(20,min_periods=16).sum(); prior=r.shift(20).rolling(40,min_periods=30).sum(); vol=r.rolling(60,min_periods=45).std()
sig=((recent-prior)/(vol+1e-12)).shift(1)
print('candidate momentum_acceleration_20_40_vol60; dates',len(px),'assets',len(W))
print('coverage',round(sig.notna().mean().mean(),4),'meanN',round(sig.notna().sum(axis=1).mean(),2),'turn10',round(sig.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4))
for h in [1,5,10,20]:
 fw=px.shift(-h)/px-1; vals=[];ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(vals); print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for label,lo,hi in [('2024-27','2024','2027-12-31'),('2028-30','2028','2030-12-31'),('2031-33','2031','2033-04-27')]:
 vals=[];fw=px.shift(-1)/px-1
 for dt in sig.loc[lo:hi].index:
  z=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(vals); print('REG1',label,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
print('If gate passes, exact library signal correlation audit is required before admission.')
