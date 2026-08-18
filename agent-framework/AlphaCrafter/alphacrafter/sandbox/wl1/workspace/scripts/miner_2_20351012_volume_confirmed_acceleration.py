import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}; V={}
for s in U:
    x=get_stock_daily_data(s,4000)
    if x is None: x=get_index_daily_data(s,4000)
    if x is not None:
        x=x.copy(); x.date=pd.to_datetime(x.date)
        P[s]=x.set_index('date').close.astype(float)
        if 'volume' in x: V[s]=pd.to_numeric(x.set_index('date').volume,errors='coerce')
p=pd.DataFrame(P).sort_index().ffill(); d=p.pct_change(); r20=p.pct_change(20); r60=p.pct_change(60)
acc=(r20-r20.median(axis=1).values[:,None])-(r60-r60.median(axis=1).values[:,None])
vol=pd.DataFrame(V).reindex(p.index).replace([np.inf,-np.inf],np.nan)
# Volume confirmation: reward acceleration when recent volume is above its causal 60d baseline.
vr=(vol.rolling(20,min_periods=10).mean()/(vol.rolling(60,min_periods=30).mean()+1e-12)).clip(.5,2.5)
vr=vr.where(vol.notna() & (vol>0),1.0)
sig=(acc*vr).shift(1)
rows=[]
for dt in sig.index:
    z=pd.concat([sig.loc[dt],p.shift(-10).loc[dt]/p.loc[dt]-1],axis=1).dropna()
    if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=q.replace([np.inf,-np.inf],np.nan).dropna()
print('dates',len(q),'avgN',q.n.mean(),'coverage',q.n.mean()/15)
print('IC10',q.ic.mean(),'ICIRdaily',q.ic.mean()/q.ic.std(),'hit',(q.ic>0).mean())
print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for h in [5,10,20,40]:
 a=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],p.shift(-h).loc[dt]/p.loc[dt]-1],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('decay',h,np.nanmean(a))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
 z=q.loc[a:b,'ic']; print('regime',a,b,len(z),z.mean(),z.mean()/z.std() if z.std()>0 else np.nan)
q.to_csv('scripts/miner_2_20351012_volume_confirmed_acceleration_ic.csv')
sig.to_csv('scripts/miner_2_20351012_volume_confirmed_acceleration_signal.csv',index_label='date')
