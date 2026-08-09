import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2026-12-31');P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut];P[s]=d.close
p=pd.concat(P,axis=1);r=p.pct_change(); bm=r.mean(axis=1)
for w in [20,60,120]:
 f=pd.DataFrame(index=r.index,columns=U,dtype=float)
 for i in range(w,len(r)):
  for s in U:
   x=r[s].iloc[i-w:i]; z=bm.iloc[i-w:i];
   q=pd.concat([x,z],axis=1).dropna()
   if len(q)>=max(8,w//2) and q.iloc[:,1].var()>0:
    beta=q.iloc[:,0].cov(q.iloc[:,1])/q.iloc[:,1].var();f.loc[f.index[i],s]=(x.iloc[-1]*0+((q.iloc[:,0]-beta*q.iloc[:,1]).add(1).prod()-1))
 a=[];ds=[];ns=[]
 for i in range(len(r)-1):
  q=pd.concat([f.iloc[i],r.iloc[i+1].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:a.append(spearmanr(q.iloc[:,0],q.y).statistic);ds.append(r.index[i]);ns.append(len(q))
 a=np.array(a);print('W',w,'dates',len(a),'N',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4),'turn',round(np.nanmean(np.abs(f.rank(pct=True).diff()).mean(axis=1)),4))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  z=a[(pd.DatetimeIndex(ds).year>=lo)&(pd.DatetimeIndex(ds).year<=hi)];print(' reg',lo,hi,'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),4),'n',len(z))
