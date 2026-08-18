import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,4000)
 if x is None:x=get_index_daily_data(s,4000)
 if x is not None:
  x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r20=p.pct_change(20);r60=p.pct_change(60)
d=get_index_daily_data('DXY',4000)
if d is None:d=get_stock_daily_data('DXY',4000)
d=d.copy();d.date=pd.to_datetime(d.date);d=d.set_index('date').close.astype(float).reindex(p.index).ffill()
# Dollar trend regime: lagged cross-asset acceleration is damped in strong-dollar
# regimes and amplified when dollar trend is weakening.
dret=d.pct_change(60); dm=dret.rolling(120,min_periods=60).median(); mult=(1-(dret-dm)*3).clip(.65,1.35)
acc=(r20-r20.median(axis=1).values[:,None])-(r60-r60.median(axis=1).values[:,None])
sig=(acc*mult.values[:,None]).shift(1);rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],p.shift(-10).loc[dt]/p.loc[dt]-1],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
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
 z=q.loc[a:b,'ic'];print('regime',a,b,'n',len(z),'ic',z.mean(),'icir',z.mean()/z.std())
q.to_csv('scripts/miner_3_20350622_dxy_conditioned_accel_ic.csv');sig.to_csv('scripts/miner_3_20350622_dxy_conditioned_accel_signal.csv',index_label='date')
