import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for a in A:
 try:
  x=get_stock_daily_data(a,days=2200); D[a]=x.set_index('date').close.astype(float)
 except: pass
p=pd.concat(D,axis=1).sort_index().ffill(); r=p.pct_change()
# Trend continuation conditioned on volatility contraction: medium trend, rewarded when short vol is below its long baseline.
vol20=r.rolling(20).std(); vol60=r.rolling(60).std(); f=(p/p.shift(20)-1)*(vol60/vol20).clip(0.25,4)
print('dates',p.index.min(),p.index.max(),'assets',len(D))
for h in [1,5,10,20]:
 v=[];ds=[];nn=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i],(p.iloc[i+h]/p.iloc[i]-1)],axis=1).dropna()
  if len(q)>=8:v.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'));ds.append(p.index[i]);nn.append(len(q))
 s=pd.Series(v,index=ds).dropna();print('h',h,'n',len(s),'avgN',np.mean(nn),'cov',np.mean(nn)/15,'IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',(s>0).mean())
 for lab,mask in [('recent252',np.arange(len(s))>=max(0,len(s)-252)),('online2026',s.index>=pd.Timestamp('2026-07-16')),('2027',s.index>=pd.Timestamp('2027-01-01'))]:
  z=s[mask]
  if len(z):print(lab,len(z),z.mean(),z.mean()/z.std(),(z>0).mean())
print('turn',f.rank(pct=True).diff().abs().mean(axis=1).mean())