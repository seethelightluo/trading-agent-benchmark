import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-30')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut] for s in U}
p=pd.concat(P,axis=1).sort_index(); r=p.pct_change()
# Recovery from drawdown: fade assets furthest below their trailing high.
for w in [20,60,120,250]:
 f=p/p.rolling(w).max()-1
 vals=[];ns=[];ds=[]
 for i in range(len(r)-1):
  q=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   vals.append(spearmanr(q.f,q.y).statistic);ns.append(len(q));ds.append(r.index[i])
 a=np.asarray(vals);d=pd.DatetimeIndex(ds)
 print('CFG',w,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4),'years',[(y,round(a[d.year==y].mean(),5),int((d.year==y).sum())) for y in range(2020,2027) if (d.year==y).any()])
