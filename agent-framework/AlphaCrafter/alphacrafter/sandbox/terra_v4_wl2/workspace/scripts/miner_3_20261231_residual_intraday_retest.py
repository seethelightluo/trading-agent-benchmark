import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-31')
O={};C={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]; O[s]=d.open; C[s]=d.close
op=pd.concat(O,axis=1,sort=False).reindex(columns=U); cl=pd.concat(C,axis=1,sort=False).reindex(columns=U)
intr=cl/op-1; f=-(intr-intr.mean(axis=1).values[:,None]); y=cl.pct_change().shift(-1)
vals=[];dates=[];ns=[]
for i in range(len(cl)-1):
 q=pd.concat([pd.Series(f.iloc[i,:].values,index=U,name='f'),y.iloc[i].rename('y')],axis=1).dropna()
 if len(q)>=8 and q.f.nunique()>1:
  r=spearmanr(q.f,q.y).statistic
  if np.isfinite(r): vals.append(r);dates.append(cl.index[i]);ns.append(len(q))
a=np.array(vals);dt=np.array(dates); print('baseline residual intraday 1d CURRENT','dates',len(a),'N',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4),'coverage',round(np.isfinite(f).mean(),5))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31')]:
 z=a[(dt>=pd.Timestamp(lo))&(dt<=pd.Timestamp(hi))];print(lo,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
print('turnover',round(np.nanmean(np.abs(pd.DataFrame(f,index=cl.index,columns=U).rank(pct=True).diff()).mean(axis=1)),5))
