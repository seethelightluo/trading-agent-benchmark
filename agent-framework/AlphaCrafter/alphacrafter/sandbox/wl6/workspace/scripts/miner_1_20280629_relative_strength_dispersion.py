import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; prices={}
for s in U:
 try: d=get_stock_daily_data(s,days=4000)
 except Exception:
  try: d=get_index_daily_data(s,days=4000)
  except Exception: d=None
 if d is not None and len(d): prices[s]=d.set_index('date')['close'].astype(float)
print('loaded',len(prices),sorted(prices)); p=pd.DataFrame(prices).sort_index().ffill(); r=np.log(p).diff(); raw=np.log(p/p.shift(20)); down=r.where(r<0).rolling(20,min_periods=10).std(); f=(raw.sub(raw.median(axis=1),axis=0)/down.replace(0,np.nan)).shift(1); fwd=np.log(p.shift(-1)/p)
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
ser=pd.Series([x[2] for x in rows],index=pd.to_datetime([x[0] for x in rows])); print('dates',len(ser),'avg_n',np.mean([x[1] for x in rows]),'coverage',f.stack().count()/(len(f)*len(U))); print('daily_ic %.8f icir %.8f hit %.5f turnover %.8f'%(ser.mean(),ser.mean()/ser.std(),(ser>0).mean(),f.rank(axis=1).diff().abs().mean().mean()))
for h in [5,10]:
 yy=np.log(p.shift(-h)/p); vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('%dd_ic %.8f'%(h,np.nanmean(vals)))
for a,b in [('2020-01-01','2023-01-01'),('2023-01-01','2025-01-01'),('2025-01-01','2027-01-01'),('2027-01-01','2030-01-01')]:
 q=ser[(ser.index>=a)&(ser.index<b)]; print(a[:4]+'-'+b[:4],q.mean(),len(q))
