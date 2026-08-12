import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0: d=get_index_daily_data(s,5000)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); P[s]=d.drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(P).sort_index().ffill(); r=np.log(p).diff()
# Novel interpretable factor: trend persistence. Reward aligned 10/30/60-day returns,
# but scale by path efficiency so noisy trends are discounted. Lag one day.
a=np.log(p).diff(10); b=np.log(p).diff(30); c=np.log(p).diff(60)
eff=np.log(p).diff(30).abs()/r.abs().rolling(30,min_periods=20).sum()
vol=r.rolling(30,min_periods=20).std()
raw=(.30*a+.45*b+.25*c)*eff/vol.replace(0,np.nan)
sig=raw.shift(1).rank(axis=1,pct=True)
rows=[]
for dt in sig.index:
 i=p.index.get_loc(dt)
 for h in [1,5,10,20]:
  if i+h>=len(p): continue
  y=p.iloc[i+h]/p.iloc[i]-1
  z=pd.concat([sig.loc[dt].rename('x'),y.rename('y')],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,z.x.corr(z.y,method='spearman'),len(z)))
o=pd.DataFrame(rows,columns=['date','h','ic','n']); o.date=pd.to_datetime(o.date)
print('dates',o.date.nunique(),'assets',p.shape[1],'avgN',o.groupby('date').n.first().mean(),'coverage',sig.notna().mean().mean(),'turnover',sig.diff().abs().mean().mean())
for h in [1,5,10,20]:
 q=o[o.h==h].groupby('date').ic.first(); print('h',h,'obs',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
for a0,b0 in [(2020,2022),(2023,2025),(2026,2028),(2029,2030),(2031,2031)]:
 g=o[(o.h==20)&(o.date.dt.year>=a0)&(o.date.dt.year<=b0)]
 if len(g): print('period',a0,b0,'IC %.6f ICIR %.6f n=%d'%(g.ic.mean(),g.ic.mean()/g.ic.std(ddof=1),len(g)))
sig.to_csv('scripts/miner_1_20310529_trend_persistence_quality_signal.csv')
