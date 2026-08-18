import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 d=get_stock_daily_data(s,2500)
 if d is not None:
  d=d.copy();d.date=pd.to_datetime(d.date);D[s]=d.drop_duplicates('date').set_index('date').sort_index()
px=pd.DataFrame({s:x.close for s,x in D.items()}).sort_index().ffill();r=px.pct_change()
# acceleration: recent 5d return relative to prior 15d average daily return
f=(px.pct_change(5)-px.pct_change(20)*.25)
for h in [1,5,10]:
 q=[];dates=[];nn=[]
 for i in range(len(r.index)-h):
  z=pd.concat([f.iloc[i],r.iloc[i+1:i+h+1].sum()],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));dates.append(r.index[i]);nn.append(len(z))
 q=pd.Series(q,index=dates).dropna();print('h',h,'dates',len(q),'avgN',np.mean(nn),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
 if h==1:
  for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31')]:
   x=q.loc[a:b];print(a,len(x),x.mean(),x.mean()/x.std(ddof=1))
