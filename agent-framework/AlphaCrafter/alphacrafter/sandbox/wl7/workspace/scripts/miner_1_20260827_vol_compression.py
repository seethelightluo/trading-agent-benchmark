import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_account_dict
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 try:d=get_stock_daily_data(s,2500)
 except Exception as e: print('skip',s,e); continue
 if d is not None and len(d):
  d=d.copy(); d['date']=pd.to_datetime(d['date']); D[s]=d.drop_duplicates('date').set_index('date').sort_index()
print('assets',len(D), {s:len(x) for s,x in D.items()})
px=pd.DataFrame({s:x['close'] for s,x in D.items()}).sort_index().ffill()
rets=px.pct_change()
f=-(rets.rolling(5,min_periods=5).std()/rets.rolling(60,min_periods=40).std())
ics=[]; dates=[]; ns=[]; turns=[]; prev=None
for i in range(len(rets.index)-1):
 z=pd.concat([f.iloc[i],rets.iloc[i+1]],axis=1).dropna()
 if len(z)>=8:
  q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman'); ics.append(q);dates.append(rets.index[i]);ns.append(len(z))
  r=f.iloc[i].rank(pct=True)
  if prev is not None:turns.append((r-prev).abs().mean())
  prev=r
x=pd.Series(ics,index=pd.to_datetime(dates)).dropna(); print('dates',len(x),'avgN',np.mean(ns),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean(),'turn',np.mean(turns))
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31')]:
 q=x.loc[a:b]; print(a,len(q),q.mean(),q.mean()/q.std(ddof=1))
for h in [5,10]:
 q=[]
 for i in range(len(rets.index)-h):
  z=pd.concat([f.iloc[i],rets.iloc[i+1:i+h+1].sum()],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(q).dropna();print('h',h,len(q),q.mean(),q.mean()/q.std(ddof=1))
