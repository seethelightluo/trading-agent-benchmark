import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0: d=get_index_daily_data(s,5000)
 if d is not None and len(d):
  d=d.copy();d.date=pd.to_datetime(d.date);P[s]=d.drop_duplicates('date').set_index('date').close.astype(float)
pd0=pd.DataFrame(P).sort_index().ffill(); r=np.log(pd0).diff()
# Breadth-confirmed momentum: medium trend, penalized by short-term exhaustion,
# and suppressed when the asset is below its own 60-day trend.
mom20=np.log(pd0).diff(20); rev5=-np.log(pd0).diff(5)
trend=(pd0>pd0.rolling(60,min_periods=40).mean()).astype(float)
breadth=trend.mean(axis=1)
# retain a continuous breadth confirmation: weak breadth lowers trend scores,
# while the asset-level trend gate prevents unsupported leaders.
sig0=(mom20+0.35*rev5)*(0.65+0.70*breadth).values[:,None]*trend.replace(0,np.nan)
sig=sig0.shift(1).rank(axis=1,pct=True)
rows=[]
for dt in sig.index:
 if dt not in pd0.index: continue
 i=pd0.index.get_loc(dt)
 for h in [1,5,10,20]:
  if i+h>=len(pd0): continue
  y=pd0.iloc[i+h]/pd0.iloc[i]-1
  z=pd.concat([sig.loc[dt].rename('x'),y.rename('y')],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,z.x.corr(z.y,method='spearman'),len(z)))
o=pd.DataFrame(rows,columns=['date','h','ic','n']);o.date=pd.to_datetime(o.date)
print('dates',o.date.nunique(),'assets',pd0.shape[1],'avgN',o.groupby('date').n.first().mean(),'coverage',sig.notna().mean().mean(),'turnover',sig.diff().abs().mean().mean())
for h in [1,5,10,20]:
 q=o[o.h==h].groupby('date').ic.first();print('h',h,'obs',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
for yr,g in o[o.h==10].groupby(o[o.h==10].date.dt.year): print('year',yr,'IC %.6f ICIR %.6f n=%d'%(g.ic.mean(),g.ic.mean()/g.ic.std(ddof=1),len(g)))
sig.to_csv('scripts/miner_3_20310403_breadth_confirmed_momentum_signal.csv')
