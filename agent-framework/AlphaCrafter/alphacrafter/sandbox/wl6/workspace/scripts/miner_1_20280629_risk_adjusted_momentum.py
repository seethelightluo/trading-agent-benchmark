import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 try:d=get_stock_daily_data(s,days=4000)
 except Exception:
  try:d=get_index_daily_data(s,days=4000)
  except Exception:d=None
 if d is not None and len(d):P[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(P).sort_index().ffill(); r=np.log(p).diff(); f=(np.log(p/p.shift(40))/r.rolling(40,min_periods=20).std()).shift(1); y=np.log(p.shift(-1)/p); out=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.nunique().min()>1:out.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
s=pd.Series([x[2] for x in out],index=pd.to_datetime([x[0] for x in out]));print('dates',len(s),'avg_n',np.mean([x[1] for x in out]),'coverage',f.stack().count()/(len(f)*15));print('ic',s.mean(),'icir',s.mean()/s.std(),'hit',(s>0).mean(),'turn',f.rank(axis=1).diff().abs().mean().mean())
for h in [5,10]:
 yy=np.log(p.shift(-h)/p);zv=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.nunique().min()>1:zv.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print(h,np.nanmean(zv))
for a,b in [('2020','2023'),('2023','2025'),('2025','2027'),('2027','2030')]:
 q=s[(s.index>=a)&(s.index<b)];print(a,q.mean(),len(q))
