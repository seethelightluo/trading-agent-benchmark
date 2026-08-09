import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-30')
# Date-aligned per asset; downside-risk adjusted reversal, with long history.
P={}; R={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]
 P[s]=x.close; R[s]=x.close.pct_change()
p=pd.concat(P,axis=1).sort_index(); r=pd.concat(R,axis=1).reindex(p.index)
for h,w in [(3,20),(5,20),(5,60),(10,60),(5,120)]:
 dn=r.where(r<0).rolling(w,min_periods=max(10,w//2)).std()
 f=-r.rolling(h).sum()/dn
 vals=[]; ns=[]; ds=[]
 for i in range(len(r)-1):
  q=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   vals.append(spearmanr(q.f,q.y).statistic); ns.append(len(q)); ds.append(r.index[i])
 a=np.asarray(vals); d=pd.DatetimeIndex(ds)
 print('CFG',h,w,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4),'years',[(y,round(a[d.year==y].mean(),5),int((d.year==y).sum())) for y in range(2020,2027) if (d.year==y).any()])
