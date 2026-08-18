import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is None or len(x)==0:x=get_index_daily_data(s,days=5000)
 if x is not None and len(x):D[s]=x.assign(date=pd.to_datetime(x.date)).set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change();v=pd.read_csv('../persistent/index_data/VIX.csv');v.date=pd.to_datetime(v.date);v=v.set_index('date').close.reindex(p.index).ffill();gate=(v>v.rolling(120,min_periods=60).quantile(.65)).astype(float)
down=r.clip(upper=0).rolling(20).std();f=(-r.rolling(5).sum()).div(down+1e-8).mul(gate,axis=0).shift(1);y=r.shift(-1);rows=[]
for dt in f.index:
 a=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(a)>=8:rows.append((dt,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');print('assets',len(D),'dates',len(z),'avgN',z.n.mean(),'coverage',z.n.mean()/15);print('dailyIC %.8f dailyICIR %.8f hit %.4f turnover %.4f'%(z.ic.mean(),z.ic.mean()/z.ic.std(),(z.ic>0).mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for label,lo,hi in [('2020-23','2020-01-01','2023-12-31'),('2024-26','2024-01-01','2026-12-31'),('2027-29','2027-01-01','2029-12-31'),('2030-32','2030-01-01','2032-12-31'),('2033-34','2033-01-01','2034-12-31')]:
 q=z.loc[lo:hi].ic;print(label,len(q),q.mean(),q.mean()/q.std() if len(q)>1 else np.nan)
for h in [5,10,20,40]:
 fw=p.shift(-h)/p-1; rr=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(a)>=8:rr.append(a.iloc[:,0].corr(a.iloc[:,1]))
 rr=pd.Series(rr).dropna();print('decay',h,rr.mean(),rr.mean()/rr.std())
f.to_csv('scripts/miner_2_20340609_dispersion_reversal_signal.csv')
