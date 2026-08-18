import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x):D[s]=x.assign(date=pd.to_datetime(x.date)).set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change();
v=pd.read_csv('../persistent/index_data/VIX.csv');v.date=pd.to_datetime(v.date);v=v.set_index('date').close.reindex(p.index).ffill(); med=v.rolling(120,min_periods=60).median(); vs=((v-med)/(v.rolling(120,min_periods=60).std()+1e-8)).clip(lower=0,upper=3)
# Combined macro stress and cross-sectional dispersion, continuously scaled.
disp=r.rolling(20).std().mean(axis=1); ds=(disp-disp.rolling(120,min_periods=60).median())/(disp.rolling(120,min_periods=60).std()+1e-8); intensity=(vs.clip(lower=0)+ds.clip(lower=0)).clip(upper=5)
down=r.clip(upper=0).rolling(20).std();f=(-r.rolling(5).sum()).div(down+1e-8).mul(intensity,axis=0).shift(1);y=p.pct_change(10).shift(-10);rows=[]
for dt in f.index:
 a=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(a)>=8:rows.append((dt,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');mu=z.ic.mean();sd=z.ic.std();print('assets',len(D),'dates',len(z),'avgN',z.n.mean(),'coverage',z.n.mean()/15);print('IC %.8f ICIR %.8f hit %.4f turnover %.4f'%(mu,mu/sd,(z.ic>0).mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for label,lo,hi in [('2024-26','2024-01-01','2026-12-31'),('2027-29','2027-01-01','2029-12-31'),('2030-32','2030-01-01','2032-12-31'),('2033-34','2033-01-01','2034-12-31')]:
 q=z.loc[lo:hi].ic;print(label,len(q),q.mean(),q.mean()/q.std())
for h in [5,10,20]:
 yy=p.pct_change(h).shift(-h);q=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(a)>=8:q.append(a.iloc[:,0].corr(a.iloc[:,1]))
 print('decay',h,np.nanmean(q),np.nanmean(q)/np.nanstd(q))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20340106_stress_dispersion_reversal_signal.csv',index=False)
