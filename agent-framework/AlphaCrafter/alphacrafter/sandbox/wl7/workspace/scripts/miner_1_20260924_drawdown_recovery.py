import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,2500)
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); D[s]=d.drop_duplicates('date').set_index('date').sort_index()
px=pd.DataFrame({s:x.close for s,x in D.items()}).sort_index().ffill()
# Drawdown recovery: current price relative to 120d high, normalized by recent 20d volatility.
r=px.pct_change(); dd=px/px.rolling(120,min_periods=60).max()-1
vol=r.rolling(20,min_periods=15).std(); f=dd/(vol*np.sqrt(20))
for h in [1,5,10]:
 qs=[]; ds=[]; ns=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i],r.iloc[i+1:i+h+1].sum()],axis=1).dropna()
  if len(z)>=8:
   qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ds.append(px.index[i]);ns.append(len(z))
 q=pd.Series(qs,index=ds).dropna(); print('h',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
 if h==1:
  for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31')]:
   x=q.loc[a:b]; print('regime',a,'dates',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1))
# mean daily turnover of rank signal
rank=f.rank(axis=1,pct=True); print('coverage',f.notna().mean().mean(),'turnover',rank.diff().abs().mean().mean())
