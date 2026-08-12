import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
xs=[]
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0: d=get_index_daily_data(s,5000)
 q=d.copy(); q.date=pd.to_datetime(q.date)
 xs.append(q.drop_duplicates('date').set_index('date').close.rename(s))
p=pd.concat(xs,axis=1).sort_index().ffill(); r=p.pct_change()
# Candidate: short-term reversal, scaled by medium realized volatility. Lagged one completed day.
vol=r.rolling(30,min_periods=20).std()*np.sqrt(30)
raw=-(p.pct_change(5))/(vol+1e-8)
sig=raw.shift(1)
rows=[]
for i,dt in enumerate(p.index[:-21]):
 x=sig.loc[dt]
 for h in [1,5,10,20]:
  y=p.iloc[i+h]/p.iloc[i]-1
  z=pd.concat([x.rename('x'),y.rename('y')],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,len(z),z.x.corr(z.y,method='spearman')))
o=pd.DataFrame(rows,columns=['date','h','n','ic'])
print('range',p.index.min(),p.index.max(),'assets',p.shape[1],'dates',o.date.nunique(),'avgN',o.n.mean())
for h in [1,5,10,20]:
 q=o[o.h==h].groupby('date').ic.first(); print('h',h,'obs',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
for y,g in o[o.h==10].groupby(o[o.h==10].date.dt.year):
 q=g.groupby('date').ic.first()
 if len(q)>5: print('yr',y,'obs',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage %.4f turnover %.6f'%(sig.notna().mean().mean(),sig.rank(pct=True).diff().abs().mean().mean()))
sig.to_csv('scripts/miner_2_20301114_short_reversal_signal.csv')
