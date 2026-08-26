import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is not None and len(d):
  x=d[['date','close']].copy();x.date=pd.to_datetime(x.date);px[s]=x.drop_duplicates('date').set_index('date').close.sort_index()
p=pd.DataFrame(px).sort_index().ffill();r=p.pct_change();v=r.rolling(20,min_periods=15).std()*np.sqrt(20)
# Contrarian short-horizon return, normalized by risk
sig=-(p.pct_change(10)/(v+1e-8))
for h in [5,10,20]:
 y=p.shift(-h)/p-1; a=[];n=[];ds=[]
 for dt in sig.index:
  q=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   c=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
   if np.isfinite(c):a.append(c);n.append(len(q));ds.append(dt)
 a=pd.Series(a,index=ds); print('H',h,'dates',len(a),'meanN',np.mean(n),'coverage',np.mean(n)/15,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
 for st in ['2026-07-16','2028-01-01','2029-01-01']:
  z=a[a.index>=st];print('PER',st,len(z),z.mean(),z.mean()/z.std(ddof=1),np.mean(z>0))
