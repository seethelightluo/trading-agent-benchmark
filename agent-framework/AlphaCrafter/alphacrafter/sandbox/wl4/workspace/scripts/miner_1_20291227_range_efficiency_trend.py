import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2029-12-26'); base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv')); d.date=pd.to_datetime(d.date); d=d[d.date<=cut].set_index('date').sort_index(); px[s]=d.close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); lr=np.log1p(r)
# Persistent directional movement: net 10d log return divided by path length, risk adjusted; lagged.
net=lr.rolling(10,min_periods=8).sum(); path=lr.abs().rolling(10,min_periods=8).sum(); vol=lr.rolling(20,min_periods=15).std()
sig=(net/(path+1e-9)/(vol+1e-9)).shift(1)
print('rows',len(P),'dates',P.index.min().date(),P.index.max().date(),'cutoff',cut.date())
for h in [1,5,10,20]:
 fwd=P.shift(-h)/P-1; vals=[]; ns=[]
 for dt in P.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append(q); ns.append(len(z))
 a=np.asarray(vals); ic=a.mean(); ir=ic/(a.std(ddof=1)+1e-12)*np.sqrt(len(a))
 print(h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round(np.mean(a>0),4))
 for n in [250,500]:
  if len(a)>=n:
   q=a[-n:]; print('recent',n,'IC',round(q.mean(),6),'ICIR',round(q.mean()/(q.std(ddof=1)+1e-12)*np.sqrt(n),6))
print('turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6),'coverage',round(sig.notna().sum().sum()/sig.size,4))
