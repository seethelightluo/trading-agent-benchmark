import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2026-12-03')
O={}; C={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]
 O[s]=d.open;C[s]=d.close
op=pd.concat(O,axis=1).reindex(columns=U);cl=pd.concat(C,axis=1).reindex(columns=U); intr=cl/op-1
# prior completed-day intraday reversal; test smoothing and volatility normalization
for w in [1,3,5,10,20]:
 for norm in [False,True]:
  f=-intr.rolling(w,min_periods=w).mean()
  if norm:
   f=f/(intr.rolling(20,min_periods=10).std()+1e-8)
  vals=[];dates=[];ns=[]
  for i in range(len(cl)-1):
   q=pd.concat([f.iloc[i],(cl.pct_change().iloc[i+1]).rename('y')],axis=1).dropna()
   if len(q)>=8 and q.iloc[:,0].nunique()>1:
    vals.append(spearmanr(q.iloc[:,0],q.y).statistic);dates.append(cl.index[i]);ns.append(len(q))
  a=np.array(vals); z=a/a.std(ddof=1)
  turn=np.nanmean(np.abs(f.rank(pct=True).diff()).mean(axis=1))
  print('w',w,'norm',norm,'dates',len(a),'N',round(np.mean(ns),2),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round(np.mean(a>0),4),'turn',round(turn,4))
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31')]:
   aa=a[(np.array(dates)>=pd.Timestamp(lo))&(np.array(dates)<=pd.Timestamp(hi))]
   if len(aa): print(' regime',lo,'n',len(aa),'ic',round(aa.mean(),5),'icir',round(aa.mean()/aa.std(ddof=1),5))
