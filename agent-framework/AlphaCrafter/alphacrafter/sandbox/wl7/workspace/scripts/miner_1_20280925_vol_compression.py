import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-09-25')
P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date'); P[s]=d.close[d.index<=end]
px=pd.DataFrame(P).sort_index().ffill(); r=px.pct_change()
# Volatility compression: lagged medium-term realized volatility decline, cross-sectionally favor assets whose 20d risk is low versus 60d risk.
v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std()
sig=(-(v20/(v60+1e-8)-1)).shift(1)
fwd=px.shift(-10)/px-1
z=[];ns=[];ds=[]
for dt in sig.index:
 a=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(a)>=8:
  q=spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic
  if np.isfinite(q): z.append(q);ns.append(len(a));ds.append(dt)
z=np.asarray(z);print('DATES',len(z),'AVG_N',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'HIT',round((z>0).mean(),4))
print('TURN',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6),'COVERAGE',round(sig.notna().sum().sum()/sig.size,4))
for lab,a,b in [('2020-22','2020-01-01','2022-12-31'),('2023-25','2023-01-01','2025-12-31'),('2026-28','2026-01-01','2028-09-25')]:
 q=z[[a<=str(d.date())<=b for d in ds]];print('REG',lab,'N',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
sig.index.name='date';sig.to_csv('scripts/miner_1_20280925_vol_compression_signal.csv')
for h in [1,5,20]:
 f=px.shift(-h)/px-1; zz=[]
 for dt in sig.index:
  a=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   q=spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic
   if np.isfinite(q):zz.append(q)
 print('DECAY',h,round(np.mean(zz),6),round(np.mean(zz)/np.std(zz,ddof=1),6),len(zz))
print('RANGE',px.index.min().date(),px.index.max().date())
