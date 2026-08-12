import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0:d=get_index_daily_data(s,5000)
 if d is not None and len(d):
  d=d.copy();d.date=pd.to_datetime(d.date);P[s]=d.drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(P).sort_index().ffill(); r=np.log(p).diff(); v20=r.rolling(20,min_periods=10).std()
vix=pd.read_csv('../persistent/index_data/VIX.csv');vix.date=pd.to_datetime(vix.date); v=vix.set_index('date').close.reindex(p.index).ffill()
# Contrarian short-horizon reversal, enabled only in elevated volatility regimes.
shock=(v/(v.rolling(120,min_periods=40).median()+1e-9)).clip(.7,2.0)
sig=(-np.log(p).diff(5)/(v20+1e-8)).mul((shock-0.8).clip(0,1.2),axis=0).shift(1).rank(axis=1,pct=True)
rows=[]
for dt in sig.index:
 i=p.index.get_loc(dt)
 for h in [1,5,10,20]:
  if i+h>=len(p):continue
  z=pd.concat([sig.loc[dt].rename('x'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8:rows.append((dt,h,z.x.corr(z.y,method='spearman'),len(z)))
o=pd.DataFrame(rows,columns=['date','h','ic','n']);print('dates',o.date.nunique(),'assets',p.shape[1],'avgN',o.groupby('date').n.first().mean(),'coverage',sig.notna().mean().mean(),'turnover',sig.diff().abs().mean().mean())
for h in [1,5,10,20]:
 q=o[o.h==h].groupby('date').ic.first();print('h',h,'obs',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
for a,b in [(2020,2022),(2023,2025),(2026,2028),(2029,2030),(2031,2031)]:
 g=o[(o.h==5)&(o.date.dt.year>=a)&(o.date.dt.year<=b)]
 if len(g):print('period',a,b,'IC %.6f ICIR %.6f n=%d'%(g.ic.mean(),g.ic.mean()/g.ic.std(ddof=1),len(g)))
sig.to_csv('scripts/miner_2_20310807_vix_elevated_reversal_signal.csv')
