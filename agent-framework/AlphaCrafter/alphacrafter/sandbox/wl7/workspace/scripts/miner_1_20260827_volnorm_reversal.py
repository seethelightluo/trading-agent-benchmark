import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,2500)
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); D[s]=d.drop_duplicates('date').set_index('date').sort_index()
px=pd.DataFrame({s:x.close for s,x in D.items()}).sort_index().ffill(); ret=px.pct_change(); vol=ret.rolling(20).std()
# volatility-normalized short-horizon reversal; high recent losses imply positive signal
f=-ret.rolling(5).sum()/vol
for h in [1,5,10]:
 ic=[]; ds=[]; ns=[]
 for i in range(len(px)-h):
  fr=ret.iloc[i+1:i+h+1].sum()
  z=pd.concat([f.iloc[i],fr],axis=1).dropna()
  if len(z)>=8: ic.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ds.append(px.index[i]);ns.append(len(z))
 q=pd.Series(ic,index=ds).dropna(); print('h',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if h==1:
  for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31')]:
   x=q.loc[a:b]; print('regime',a,'dates',len(x),'ICIR',round(x.mean()/x.std(ddof=1),6),'IC',round(x.mean(),6))
print('turnover_proxy',f.rank(axis=1,pct=True).diff().abs().mean().mean(),'coverage',f.notna().mean().mean(),'assets',len(D))
