import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0:d=get_index_daily_data(s,5000)
 if d is not None and len(d):P[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
px=pd.DataFrame(P).sort_index().ffill(); r=px.pct_change()
# Trend persistence: directional displacement divided by realized path length; rewards smooth trends.
f=(np.log(px/px.shift(40))/(r.abs().rolling(40,min_periods=30).sum()+1e-9)).shift(1)
y=px.pct_change(10).shift(-10); ics=[];ds=[];ns=[]
for dt in f.index:
 a=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(a)>=8:
  c=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
  if np.isfinite(c):ics.append(c);ds.append(dt);ns.append(len(a))
z=np.array(ics); print('dates',len(z),'avgN',np.mean(ns),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0),'coverage',f.notna().sum(axis=1).mean()/15,'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
for lo,hi in [(2020,2023),(2024,2026),(2027,2029),(2030,2032),(2033,2034)]:
 q=np.array([v for d,v in zip(ds,ics) if lo<=d.year<=hi]);print('REG',lo,hi,'n',len(q),'IC',q.mean() if len(q) else np.nan,'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
for h in [5,10,20,40]:
 yy=px.pct_change(h).shift(-h);q=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   c=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if np.isfinite(c):q.append(c)
 print('DECAY',h,np.mean(q),len(q))
f.loc[ds].to_csv('scripts/miner_2_20340428_path_efficiency_signal.csv',index_label='date')
