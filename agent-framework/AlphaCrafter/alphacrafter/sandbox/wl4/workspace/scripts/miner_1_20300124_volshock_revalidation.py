import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2030-01-23'); b='../persistent/stock_data'; px={}
for s in U:
 d=pd.read_csv(os.path.join(b,s+'.csv'));d.date=pd.to_datetime(d.date);px[s]=d[d.date<=cut].set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index();r=P.pct_change();rv=r.rolling(20,min_periods=15).std();base=rv.rolling(120,min_periods=60).mean();shock=(rv/(base+1e-8)).clip(.5,3);sig=(-(r.rolling(5,min_periods=5).sum())/(rv*np.sqrt(20)+1e-8)*shock).shift(1)
for h in [1,5,10,20]:
 f=P.shift(-h)/P-1;a=[];ns=[]
 for dt in P.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):a.append(q);ns.append(len(z))
 a=np.array(a); print(h,'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a)),'hit',np.mean(a>0),'recent250',a[-250:].mean(),a[-250:].mean()/(a[-250:].std(ddof=1)+1e-12)*np.sqrt(250))
print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),'valid',sig.notna().sum().sum()/sig.size)
