import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2034-08-03'); P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv');d.date=pd.to_datetime(d.date);P[s]=d[d.date<=cut].set_index('date').close
P=pd.DataFrame(P).sort_index();r=P.pct_change(); f=(P.pct_change(20)/r.rolling(40).std()).shift(1); fr=P.shift(-10)/P-1
I=[]; ns=[]; cov=[]; tr=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):I.append((dt,q));ns.append(len(z));cov.append(len(z)/15)
  tr.append((f.loc[dt].rank(pct=True)-f.shift().loc[dt].rank(pct=True)).abs().mean())
I=pd.Series(dict(I));print('dates',len(I),'range',I.index.min(),I.index.max(),'avg_n',np.mean(ns),'coverage',np.mean(cov));print('IC %.6f ICIR %.6f hit %.4f'%(I.mean(),I.mean()/I.std(),(I>0).mean()));
for n in [120,260,520,780]:q=I.tail(n);print('recent',n,'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std()))
print('turnover',np.nanmean(tr))
for h in [1,5,10,20]:
 y=P.shift(-h)/P-1;a=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(a),len(a))
