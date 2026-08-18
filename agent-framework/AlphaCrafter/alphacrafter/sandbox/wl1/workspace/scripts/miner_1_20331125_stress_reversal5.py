import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x):
  x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill();v=pd.read_csv('../persistent/index_data/VIX.csv');v.date=pd.to_datetime(v.date);v=v.set_index('date').iloc[:,0].astype(float).reindex(p.index).ffill();thr=v.rolling(252,min_periods=126).quantile(.7).shift(1)
# stress-triggered short-horizon reversal, neutral otherwise via zero score
f=(-p.pct_change(5)).shift(1).where(v>thr,0);fr=p.pct_change(10).shift(-10);r=[]
for dt in f.index:
 a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(a)>=8:r.append((dt,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
z=pd.DataFrame(r,columns=['date','ic','n']).set_index('date');print('dates',len(z),'avgN',z.n.mean(),'coverage',z.n.mean()/15,'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(),'hit',(z.ic>0).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for l,a,b in [('2024-26','2024-01-01','2026-12-31'),('2027-29','2027-01-01','2029-12-31'),('2030-32','2030-01-01','2032-12-31'),('2033','2033-01-01','2033-10-26')]:
 q=z.loc[a:b].ic;print(l,len(q),q.mean(),q.mean()/q.std())
for h in [5,10,20]:
 y=p.pct_change(h).shift(-h);q=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(a)>=8:q.append(a.iloc[:,0].corr(a.iloc[:,1]))
 print('decay',h,np.nanmean(q),np.nanmean(q)/np.nanstd(q))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20331125_stress_reversal5_signal.csv',index=False)
