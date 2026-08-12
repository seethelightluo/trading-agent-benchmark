import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
xs=[]
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0: d=get_index_daily_data(s,5000)
 xs.append(d.drop_duplicates('date').set_index(pd.to_datetime(d.drop_duplicates('date').date)).close.rename(s))
p=pd.concat(xs,axis=1).sort_index().ffill(); r=p.pct_change()
# Multi-horizon trend persistence: blend normalized 20d and 60d returns,
# with a persistence confirmation from the fraction of positive sessions.
vol40=r.rolling(40,min_periods=25).std()*np.sqrt(40)
m20=(p.pct_change(20)/(vol40+1e-9)).rank(axis=1,pct=True)
m60=(p.pct_change(60)/(r.rolling(60,min_periods=40).std()*np.sqrt(60)+1e-9)).rank(axis=1,pct=True)
persist=r.gt(0).rolling(30,min_periods=20).mean().rank(axis=1,pct=True)
signal=(0.45*m20+0.35*m60+0.20*persist).shift(1)
rows=[]
for i,dt in enumerate(p.index[:-21]):
 for h in [1,5,10,20]:
  z=pd.concat([signal.iloc[i].rename('x'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,len(z),z.x.corr(z.y,method='spearman')))
o=pd.DataFrame(rows,columns=['date','h','n','ic'])
print('range',p.index.min(),p.index.max(),'assets',p.shape[1],'dates',o.date.nunique(),'avgN',o.n.mean())
for h in [1,5,10,20]:
 q=o[o.h==h].groupby('date').ic.first(); print('h',h,'obs',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
for y,g in o[o.h==1].groupby(o[o.h==1].date.dt.year):
 z=g.groupby('date').ic.first()
 if len(z)>5: print('yr',y,'obs',len(z),'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
print('coverage %.4f turnover %.6f'%(signal.notna().mean().mean(),signal.rank(axis=1,pct=True).diff().abs().mean().mean()))
signal.to_csv('scripts/miner_3_20301212_multihorizon_trend_signal.csv')
