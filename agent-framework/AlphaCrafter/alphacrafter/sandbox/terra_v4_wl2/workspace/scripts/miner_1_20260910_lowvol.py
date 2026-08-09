import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=2500)
 if d is not None:
  d=d.sort_values('date').set_index('date'); D[s]=d['close'].astype(float)
p=pd.concat(D,axis=1).sort_index(); ret=p.pct_change()
def calc(fac, start, end, h=1):
 a=[]; ns=[]; turns=[]
 for i in range(start,len(p)-h):
  if end is not None and not end[i]: continue
  y=p.iloc[i+h]/p.iloc[i]-1; z=pd.concat([fac.iloc[i],y],axis=1).dropna()
  if len(z)>=8:
   a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
   if i: 
    q=pd.concat([fac.iloc[i-1].rank(pct=True),fac.iloc[i].rank(pct=True)],axis=1).dropna(); turns.append(np.mean(np.abs(q.iloc[:,0]-q.iloc[:,1])))
 a=np.array(a); return len(a),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0),np.mean(ns),np.nanmean(turns)
print('assets',len(D),'dates',len(p),'range',p.index.min(),p.index.max())
for w in [5,10,20,40,60]:
 fac=-ret.rolling(w,min_periods=max(2,w//2)).std(); print(w,calc(fac,w,None))
for label,mask in [('2020-22',p.index<'2023-01-01'),('2023-24',(p.index>='2023-01-01')&(p.index<'2025-01-01')),('2025-26',p.index>='2025-01-01')]:
 fac=-ret.rolling(20,min_periods=10).std(); print(label,calc(fac,20,mask))
fac=-ret.rolling(20,min_periods=10).std()
for h in [1,5,10]: print('h',h,calc(fac,20,None,h))
