import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s, days=5000) for s in U}
px=pd.DataFrame({s:(d.set_index('date')['close'] if d is not None else pd.Series(dtype=float)) for s,d in D.items()}).sort_index()
r=px.pct_change(); ret=px/px.shift(60)-1
down=r.where(r<0,0.0).rolling(60,min_periods=40).std()*np.sqrt(252)
f=(ret/(down+1e-6)).clip(-10,10); f=f.sub(f.median(axis=1),axis=0)
rows=[]; dates=px.index
for i,dt in enumerate(dates):
 j=i+10
 if j>=len(dates): break
 z=pd.concat([f.iloc[i].rename('f'),(px.iloc[j]/px.iloc[i]-1).rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: rows.append((dt,len(z),z.f.corr(z.y,method='spearman')))
x=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); ics=x.ic.dropna()
print('dates',len(x),'range',x.index.min(),x.index.max(),'avg_n',x.n.mean(),'coverage',x.n.mean()/15)
print('IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1),'hit',(ics>0).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for h in [5,20]:
 rr=[]
 for i in range(len(dates)-h):
  z=pd.concat([f.iloc[i].rename('f'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: rr.append(z.f.corr(z.y,method='spearman'))
 print(h,'IC',np.nanmean(rr),'n',len(rr))
for n in [180,360]: print('recent',n,ics.tail(n).mean(),ics.tail(n).mean()/ics.tail(n).std(ddof=1))
print('last data',px.index.max())
