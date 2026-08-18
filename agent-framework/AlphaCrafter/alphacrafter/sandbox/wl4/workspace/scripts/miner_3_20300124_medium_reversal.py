import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2030-01-23'); b='../persistent/stock_data'; px={}
for s in U:
 d=pd.read_csv(os.path.join(b,s+'.csv'));d.date=pd.to_datetime(d.date);px[s]=d[d.date<=cut].set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); rv=r.rolling(20,min_periods=15).std()
# Medium horizon contrarian signal, risk scaled and lagged one session.
sig=(-(P/P.shift(20)-1)/(rv*np.sqrt(20)+1e-8)).shift(1)
print('rows',len(P),'range',P.index.min().date(),P.index.max().date(),'cut',cut.date())
for h in [1,5,10,20]:
 f=P.shift(-h)/P-1;a=[];ns=[]
 for dt in P.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):a.append(q);ns.append(len(z))
 a=np.asarray(a);ic=a.mean();ir=ic/(a.std(ddof=1)+1e-12)*np.sqrt(len(a))
 print(h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round(np.mean(a>0),4),'coverage',round(np.mean(ns)/15,4))
 if len(a)>=250:
  q=a[-250:];print('recent250',round(q.mean(),6),round(q.mean()/(q.std(ddof=1)+1e-12)*np.sqrt(250),6))
print('turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6),'panel_valid',round(sig.notna().sum().sum()/sig.size,4))
