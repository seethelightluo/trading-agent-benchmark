import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0:d=get_index_daily_data(s,5000)
 if d is not None and len(d):P[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
px=pd.DataFrame(P).sort_index().ffill();r=px.pct_change();down=r.clip(upper=0).rolling(40,min_periods=20).std()*np.sqrt(40)
f=(-(np.log(px/px.shift(60))-.70*np.log(px/px.shift(10)))/(down+.5*r.rolling(20,min_periods=15).std()*np.sqrt(20)+1e-6)).shift(1)
for h in [10,20]:
 fr=px.pct_change(h).shift(-h); rows=[]
 for dt in f.index:
  if dt.year<2030:continue
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8: rows.append((dt,a.iloc[:,0].corr(a.iloc[:,1],method='spearman'),len(a)))
 z=np.array([x[1] for x in rows]);print('h',h,'dates',len(z),'avgN',np.mean([x[2] for x in rows]),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0))
 for y in [2030,2031,2032,2033,2034]:
  q=np.array([x[1] for x in rows if x[0].year==y]);print(y,len(q),q.mean() if len(q) else np.nan,q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
