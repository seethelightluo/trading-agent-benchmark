import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 x=get_stock_daily_data(s,4000)
 if x is None:x=get_index_daily_data(s,4000)
 if x is not None:
  x=x.copy();x.date=pd.to_datetime(x.date);P[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(P).sort_index().ffill(); ret=p.pct_change(); r20=p.pct_change(20);r60=p.pct_change(60)
acc=(r20-r20.median(axis=1).values[:,None])-(r60-r60.median(axis=1).values[:,None])
# Causal path efficiency: net 20d displacement divided by sum absolute daily moves.
eff=r20.abs()/(ret.abs().rolling(20,min_periods=15).sum()+1e-12)
eff=eff.div(eff.median(axis=1),axis=0).clip(.4,2.5)
sig=(acc*eff).shift(1)
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],p.shift(-10).loc[dt]/p.loc[dt]-1],axis=1).dropna()
 if len(z)>=8:rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').replace([np.inf,-np.inf],np.nan).dropna()
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
 z=q.loc[a:b,'ic'];print('regime',a,b,len(z),z.mean(),z.mean()/z.std() if z.std()>0 else np.nan)
q.to_csv('scripts/miner_2_20351026_efficiency_acceleration_ic.csv');sig.to_csv('scripts/miner_2_20351026_efficiency_acceleration_signal.csv',index_label='date')
