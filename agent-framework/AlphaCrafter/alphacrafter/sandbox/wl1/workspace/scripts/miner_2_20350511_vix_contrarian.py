import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,4000)
 if x is None:x=get_index_daily_data(s,4000)
 if x is not None:D[s]=x.assign(date=pd.to_datetime(x.date)).set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
vix=get_index_daily_data('VIX',4000)
if vix is None: raise RuntimeError('VIX unavailable')
v=vix.assign(date=pd.to_datetime(vix.date)).set_index('date').close.astype(float).reindex(p.index).ffill()
# Independently validate sign-reversed VIX-adaptive contrarian: positive after negative momentum,
# amplified only when VIX is above its rolling median; one-day causal lag.
adj=(1+(v/v.rolling(60).median()-1).clip(lower=0,upper=1.5))
for look in [3,5,10]:
 sig=(-r.rolling(look).sum().mul(adj,axis=0)).shift(1)
 rows=[]; dates=[]; cov=[]
 for i,d in enumerate(sig.index):
  if i+10>=len(p): continue
  z=pd.concat([sig.loc[d],p.iloc[i+10]/p.iloc[i]-1],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].std()>0:
   rows.append(z.iloc[:,0].corr(z.iloc[:,1]));dates.append(d);cov.append(len(z)/15)
 a=np.array(rows); print('look',look,'dates',len(a),'avgN',np.mean(np.array(cov)*15),'coverage',np.mean(cov),'IC',a.mean(),'ICIR',a.mean()/a.std(),'hit',(a>0).mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
 for lo,hi in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
  q=a[(pd.DatetimeIndex(dates)>=pd.Timestamp(lo+'-01-01'))&(pd.DatetimeIndex(dates)<=pd.Timestamp(hi+'-12-31'))]
  print(' regime',lo,hi,'n',len(q),'ic',q.mean() if len(q) else np.nan,'icir',q.mean()/q.std() if len(q)>1 else np.nan)
 print('decay',end=' ')
 for h in [5,10,20,40]:
  rr=[]
  for i,d in enumerate(sig.index):
   if i+h>=len(p):continue
   z=pd.concat([sig.loc[d],p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].std()>0:rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
  print(h,round(np.nanmean(rr),5),end=' ')
 print()
