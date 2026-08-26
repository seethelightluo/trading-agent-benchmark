import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-09-11')
P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']);d=d[d.date<=end].set_index('date');P[s]=d.close
px=pd.DataFrame(P).sort_index().ffill();r=px.pct_change()
# 40-day momentum skipping last 5, normalized by lagged 60-day volatility.
sig=((px.shift(5)/px.shift(45)-1)/(r.rolling(60,min_periods=40).std().shift(5)*np.sqrt(252)+.01)).shift(1)
for h in [5,10,20]:
 f=px.shift(-h)/px-1;z=[];ns=[];ds=[]
 for dt in sig.index:
  a=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   q=spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic
   if np.isfinite(q):z.append(q);ns.append(len(a));ds.append(dt)
 z=np.array(z);print('H',h,'dates',len(z),'avgN',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
 if h==10:
  print('TURN',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6),'COVERAGE',round(sig.notna().sum().sum()/sig.size,4))
  for lab,a,b in [('2020-22','2020-01-01','2022-12-31'),('2023-25','2023-01-01','2025-12-31'),('2026-28','2026-01-01','2028-09-11')]:
   q=z[[a<=str(d.date())<=b for d in ds]];print('REG',lab,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
sig.index.name='date';sig.to_csv('scripts/miner_2_20280911_mom40_vol60_signal.csv')
