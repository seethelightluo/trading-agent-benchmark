import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2030-01-23');b='../persistent/stock_data';px={}
for s in U:
 d=pd.read_csv(os.path.join(b,s+'.csv'));d.date=pd.to_datetime(d.date);px[s]=d[d.date<=cut].set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index();r=P.pct_change();rv=r.rolling(20,min_periods=15).std();base=rv.rolling(120,min_periods=60).mean()
# Updated vol-shock reversal, lagged. Evaluate post-2025 and full history.
sig=(-(r.rolling(5,min_periods=5).sum())/(rv*np.sqrt(20)+1e-8)*(rv/(base+1e-8)).clip(.5,3)).shift(1)
for label,ix in [('full',P.index),('post2025',P.index[P.index>=pd.Timestamp('2025-01-01')]),('recent250',P.index[-250:])]:
 a=[];ns=[]
 for dt in ix:
  z=pd.concat([sig.loc[dt],(P.shift(-1)/P-1).loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):a.append(q);ns.append(len(z))
 a=np.asarray(a);print(label,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a)),6),'hit',round(np.mean(a>0),4))
print('turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6),'valid',round(sig.notna().sum().sum()/sig.size,4))
