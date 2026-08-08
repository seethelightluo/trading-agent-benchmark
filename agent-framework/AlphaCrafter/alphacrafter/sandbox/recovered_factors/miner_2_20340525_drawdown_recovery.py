import pandas as pd,numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
 x=pd.read_csv('../persistent/stock_data/'+a+'.csv'); x.date=pd.to_datetime(x.date); D[a]=x.set_index('date').sort_index()
idx=sorted(set.intersection(*[set(x.index) for x in D.values()])); idx=pd.DatetimeIndex(idx)
px=pd.DataFrame({a:D[a].reindex(idx).close for a in assets}).loc[:'2034-05-10']
r=px.pct_change(); vol=r.rolling(20,min_periods=15).std()
# Drawdown recovery: reward recent rebound from a 60-session trough, scaled by risk,
# while conditioning on a meaningful prior drawdown to avoid generic momentum.
low=px.rolling(60,min_periods=40).min(); dd=px/low-1
rebound=px/px.shift(10)-1
f=(rebound/(vol*np.sqrt(252)+1e-5))*((-dd).clip(0,0.30)/0.10).clip(0,3)
for h in [1,5,10,20]:
 q=[];ns=[]
 for j in range(len(px)-h):
  z=pd.concat([f.iloc[j],px.iloc[j+h]/px.iloc[j]-1],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 q=np.array(q); print('h',h,'dates',len(q),'meanN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6),'hit',round(np.mean(q>0),4))
print('coverage',round(f.notna().sum().sum()/f.size,6),'turnover',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),6))
for lo,hi in [(2020,2024),(2025,2029),(2030,2034)]:
 q=[]
 for j in range(len(px)-20):
  if lo<=px.index[j].year<=hi:
   z=pd.concat([f.iloc[j],px.iloc[j+20]/px.iloc[j]-1],axis=1).dropna()
   if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q);print('regime',lo,hi,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6),'hit',round(np.mean(q>0),4))
