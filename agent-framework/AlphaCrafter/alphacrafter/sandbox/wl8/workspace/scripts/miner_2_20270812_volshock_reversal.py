import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv'); d.date=pd.to_datetime(d.date); P[a]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
p=pd.concat(P,axis=1).sort_index(); ret=p.pct_change()
vol=ret.rolling(20,min_periods=15).std().shift(1)
raw=-ret.rolling(3,min_periods=3).sum().shift(1).div(vol)
shock=raw.abs().median(axis=1)
f=raw.mul(1+0.35*shock.clip(upper=2),axis=0)
fr=p.pct_change().shift(-1)
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
def met(q): return (len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(252), (q.ic>0).mean())
print('dates',len(r),'rows',r.n.sum(),'avg_n',r.n.mean(),'coverage',r.n.sum()/(len(r)*15)); print('all',met(r))
for label,mask in [('2020-22',(r.index.year<=2022)),('2023-25',(r.index.year>=2023)&(r.index.year<=2025)),('2026',(r.index.year==2026)),('2027',(r.index.year==2027)),('recent90',(r.index>=pd.Timestamp('2027-05-14')))]: print(label,met(r[mask]))
for h in [5]:
 frh=p.pct_change(h).shift(-h); v=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],frh.loc[dt]],axis=1).dropna()
  if len(z)>=8:v.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('h',h,'dates',len(v),'IC',np.mean(v),'ICIR',np.mean(v)/np.std(v,ddof=1)*np.sqrt(252))
