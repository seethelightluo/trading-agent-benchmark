import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is not None and len(d): P[s]=d.assign(date=pd.to_datetime(d.date)).drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(P).sort_index().ffill(); r=np.log(p).diff()
v=pd.read_csv('../persistent/index_data/VIX.csv'); v['date']=pd.to_datetime(v['date']); v=v.drop_duplicates('date').set_index('date')['close'].reindex(p.index).ffill()
# VIX-conditioned defensive relative strength: 20d return relative to cross-asset median,
# amplified in stress (VIX above its 120d median); lag all information by one day.
rel=np.log(p).diff(20).sub(np.log(p).diff(20).median(axis=1),axis=0)
stress=(v>v.rolling(120,min_periods=60).median()).astype(float)
sig=(rel*(1+stress)).shift(1).rank(axis=1,pct=True)
rows=[]
for dt in sig.index:
 i=p.index.get_loc(dt)
 for h in [1,5,10,20]:
  if i+h>=len(p): continue
  z=pd.concat([sig.loc[dt].rename('x'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,z.x.corr(z.y,method='spearman'),len(z)))
o=pd.DataFrame(rows,columns=['date','h','ic','n']); o.date=pd.to_datetime(o.date)
print('dates',o.date.nunique(),'assets',p.shape[1],'avgN',o.groupby('date').n.first().mean(),'coverage',sig.notna().mean().mean(),'turnover',sig.diff().abs().mean().mean())
for h in [1,5,10,20]:
 q=o[o.h==h].groupby('date').ic.first(); print('h',h,'obs',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
for a,b in [(2020,2022),(2023,2025),(2026,2028),(2029,2030),(2031,2031)]:
 g=o[(o.h==20)&(o.date.dt.year>=a)&(o.date.dt.year<=b)]
 if len(g): print('period',a,b,'IC %.6f ICIR %.6f n=%d'%(g.ic.mean(),g.ic.mean()/g.ic.std(ddof=1),len(g)))
