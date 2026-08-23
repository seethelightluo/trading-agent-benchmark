import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2027-02-10');F={};Y={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()['close'];d=d[d.index<=cut];r=d.pct_change(); down=(-r.clip(upper=0)).rolling(40,min_periods=25).mean(); F[s]=(d/d.shift(40)-1)/(down+1e-6);Y[s]=d.shift(-1)/d-1
F=pd.DataFrame(F);Y=pd.DataFrame(Y);A=[];ns=[];dates=[]
for dt in F.index:
 z=pd.concat([F.loc[dt],Y.loc[dt]],axis=1).dropna()
 if len(z)>=8:A.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));dates.append(dt)
A=np.array(A);rank=F.rank(axis=1,pct=True);tr=[]
for i in range(1,len(rank)):
 z=rank.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:tr.append(np.mean(abs(z.iloc[:,0]-z.iloc[:,1])))
print('candidate 40d trend/downside-loss');print('dates',len(A),'avg_names',np.mean(ns),'coverage',len(A)/len(F),'turnover',np.mean(tr));print('IC',A.mean(),'ICIR',A.mean()/A.std(),'hit',np.mean(A>0))
for h in [5,10,20]:
 q=[]
 for s in U:
  d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()['close'];d=d[d.index<=cut];
  if s not in globals():pass
  # use factor rows and forward returns assembled
  yy=d.shift(-h)/d-1
  # save
  Yh=yy.rename(s)
  if s==U[0]: Z=pd.DataFrame(Yh)
  else: Z=Z.join(Yh,how='outer')
 for dt in F.index:
  z=pd.concat([F.loc[dt],Z.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.mean(q),len(q))
for a,b in [(2020,2022),(2023,2024),(2025,2027)]:
 q=[v for dt,v in zip(dates,A) if a<=dt.year<=b];print('regime',a,b,len(q),np.mean(q))
q=[v for dt,v in zip(dates,A) if dt>=pd.Timestamp('2026-07-16')];print('online',len(q),np.mean(q),np.mean(q)/np.std(q))
out=F.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20270211_medium_trend_signal.csv',index=False);print('artifact rows',len(out))
