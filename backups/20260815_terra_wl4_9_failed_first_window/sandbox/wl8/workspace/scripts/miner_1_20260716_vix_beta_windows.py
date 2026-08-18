import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
D={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index()['close'] for s in U}
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index()['close'].pct_change()
for w in [20,40,120]:
 F={}
 for s,c in D.items():
  q=pd.concat([c.pct_change().rename('r'),v.rename('v')],axis=1).dropna(); F[s]=(-q.r.rolling(w,min_periods=max(12,w//2)).cov(q.v)/q.v.rolling(w,min_periods=max(12,w//2)).var()).rename(s)
 F=pd.DataFrame(F).loc[:end]; Y=pd.DataFrame({s:c.shift(-1)/c-1 for s,c in D.items()}).loc[:end]; a=[];ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],Y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(a); print('window',w,'dates',len(a),'meanN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'coverage>=8',(F.notna().sum(axis=1)>=8).mean())
