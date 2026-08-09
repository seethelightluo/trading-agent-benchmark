import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-31'); P={};V={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut];P[s]=d.close;V[s]=d.volume
p=pd.concat(P,axis=1); v=pd.concat(V,axis=1); r=p.pct_change()
for w in [5,20,60]:
 f=(v.rolling(w).mean()/v.rolling(120).mean()-1).replace([np.inf,-np.inf],np.nan)
 a=[];ds=[];ns=[]
 for i in range(len(r)-1):
  q=pd.concat([f.iloc[i],r.iloc[i+1].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:a.append(spearmanr(q.iloc[:,0],q.y).statistic);ds.append(r.index[i]);ns.append(len(q))
 a=np.array(a);print('W',w,'dates',len(a),'N',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4),'turn',round(np.nanmean(np.abs(f.rank(pct=True).diff()).mean(axis=1)),4))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  z=a[(pd.DatetimeIndex(ds).year>=lo)&(pd.DatetimeIndex(ds).year<=hi)];print(' reg',lo,hi,'ICIR',round(z.mean()/z.std(ddof=1),4),'n',len(z))
