import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=4600)
 if d is not None and len(d)>300:
  d=d[['date','close']].dropna().drop_duplicates('date').set_index('date')
  px[s]=d.close.astype(float)
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Relative curvature: recent 20d return pace versus 60d pace, after removing
# the contemporaneous cross-asset median return at each horizon. Contrarian,
# volatility scaled, and lagged one completed session.
med20=r.rolling(20,min_periods=15).sum().median(axis=1)
med60=r.rolling(60,min_periods=45).sum().median(axis=1)
r20=r.rolling(20,min_periods=15).sum(); r60=r.rolling(60,min_periods=45).sum()
curv=(r20-med20.to_numpy()[:,None])-(r60-med60.to_numpy()[:,None])/3
vol=r.rolling(60,min_periods=45).std()*np.sqrt(252)
f=(-curv/vol.replace(0,np.nan)).shift(1)
fr=p.shift(-60)/p-1
qs=[]; ns=[]; ds=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
  qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); ds.append(dt)
q=pd.Series(qs,index=pd.to_datetime(ds)).dropna()
print('DATA dates',len(p),'instruments',len(px),'range',p.index.min(),p.index.max())
print('H 60 dates',len(q),'avgN %.2f'%np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(),(q>0).mean()))
for a,b in [('2020-01-01','2023-12-31'),('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2030-12-31'),('2031-01-01','2032-08-18')]:
 z=q.loc[a:b]
 if len(z): print('REGIME',a[:4],len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(),(z>0).mean()))
print('COVERAGE %.6f TURNOVER %.6f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20320819_relative_curvature_residual_signal.csv',index=False)
