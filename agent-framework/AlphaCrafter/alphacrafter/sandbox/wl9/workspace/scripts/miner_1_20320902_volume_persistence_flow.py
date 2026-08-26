import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={};vol={}
for s in U:
 d=get_stock_daily_data(s,days=4600)
 if d is not None and len(d)>300:
  d=d[['date','close','volume']].dropna(subset=['close']).drop_duplicates('date').set_index('date')
  px[s]=d.close.astype(float); vol[s]=d.volume.astype(float).replace(0,np.nan)
p=pd.DataFrame(px).sort_index(); v=pd.DataFrame(vol).reindex(p.index)
r=p.pct_change()
# Flow persistence: signed return pressure accumulated over 20d, weighted by abnormal volume,
# normalized by recent volatility and lagged one completed session.
vol_z=(v/v.rolling(60,min_periods=30).median()-1).clip(-3,3)
pressure=(r*vol_z).rolling(20,min_periods=12).sum()
rv=r.rolling(60,min_periods=40).std()*np.sqrt(252)
f=(pressure/rv.replace(0,np.nan)).shift(1)
print('DATA dates',len(p),'instruments',len(px),'range',p.index.min(),p.index.max())
for h in [5,10,20,40,60]:
 fr=p.shift(-h)/p-1; qs=[]; ns=[]; ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); ds.append(dt)
 q=pd.Series(qs,index=pd.to_datetime(ds)).dropna()
 print('H',h,'dates',len(q),'avgN %.2f'%np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(),(q>0).mean()))
 if h==60:
  for a,b in [('2020-01-01','2023-12-31'),('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2030-12-31'),('2031-01-01','2032-09-01')]:
   z=q.loc[a:b]
   if len(z): print('REGIME',a[:4],len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(),(z>0).mean()))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20320902_volume_persistence_flow_signal.csv',index=False)
print('COVERAGE %.6f TURNOVER %.6f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
