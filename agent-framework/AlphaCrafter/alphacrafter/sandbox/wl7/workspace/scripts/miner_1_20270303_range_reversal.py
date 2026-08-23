import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2027-03-03'); q={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date).dt.normalize()
 q[s]=d[d.date<=cut].drop_duplicates('date').set_index('date').close.astype(float)
p=pd.concat(q,axis=1).sort_index(); r=p.pct_change()
# Range-normalized 3-day reversal. Signal is lagged one completed session.
vol=r.rolling(20,min_periods=15).std(); rng=(p.rolling(20,min_periods=15).max()-p.rolling(20,min_periods=15).min())/p
f=(-r.rolling(3,min_periods=3).sum()/(vol*np.sqrt(3)+1e-12)/(1+rng)).shift(1)
def stat(fr):
 z=[];ns=[]
 for i in range(len(p)):
  v=f.iloc[i].notna()&fr.iloc[i].notna()
  if v.sum()>=8:z.append(spearmanr(f.iloc[i][v],fr.iloc[i][v]).statistic);ns.append(v.sum())
 z=np.array(z);return len(z),np.mean(ns),z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252),np.mean(z>0)
fr1=r.shift(-1); n,av,ic,ir,hit=stat(fr1)
print('candidate=range_normalized_3d_reversal');print('dates',n,'avgN',round(av,2),'coverage',round(100*av/15,2),'IC',round(ic,8),'ICIR',round(ir,6),'hit',round(hit,4))
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
for h in [5,10,20]:
 n,av,ic,ir,hit=stat(p.pct_change(h).shift(-h));print('decay',h,'dates',n,'IC',round(ic,8),'ICIR',round(ir,6))
for a,b in [('2020','2022'),('2023','2024'),('2025','2027')]:
 z=[]
 for dt in f.loc[a:b].index:
  x=f.loc[dt];y=fr1.loc[dt];v=x.notna()&y.notna()
  if v.sum()>=8:z.append(spearmanr(x[v],y[v]).statistic)
 print('regime',a,b,'dates',len(z),'IC',round(float(np.mean(z)),8) if z else None)
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20270303_range_reversal_signal.csv',index=False)
