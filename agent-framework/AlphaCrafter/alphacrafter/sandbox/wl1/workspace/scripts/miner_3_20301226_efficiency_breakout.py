import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
close={}; high={}; low={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0: d=get_index_daily_data(s,5000)
 d=d.drop_duplicates('date');ix=pd.to_datetime(d.date)
 for dest,col in [(close,'close'),(high,'high'),(low,'low')]: dest[s]=d.set_index(ix)[col]
p=pd.DataFrame(close).sort_index().ffill(); hi=pd.DataFrame(high).reindex(p.index).ffill(); lo=pd.DataFrame(low).reindex(p.index).ffill(); r=p.pct_change()
# Lagged trend-efficiency breakout: directional displacement divided by realized path and range risk.
path=r.abs().rolling(30,min_periods=20).sum(); efficiency=(r.rolling(30,min_periods=20).sum().abs()/(path+1e-12))
atr=((hi-lo)/p).rolling(20,min_periods=15).mean(); ret=p.pct_change(20)
# signed trend strength retains direction; efficiency suppresses choppy moves, rank cross-section
signal=((ret/(atr*np.sqrt(20)+1e-9))*efficiency).rank(axis=1,pct=True).shift(1)
rows=[]
for i in range(len(p)-21):
 for h in [1,5,10,20]:
  z=pd.concat([signal.iloc[i].rename('x'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8: rows.append((p.index[i],h,z.x.corr(z.y,method='spearman'),len(z)))
o=pd.DataFrame(rows,columns=['date','h','ic','n']); print('dates',o.date.nunique(),'assets',p.shape[1],'avgN',o.groupby('date').n.first().mean(),'coverage',signal.notna().mean().mean())
for h in [1,5,10,20]:
 q=o[o.h==h].groupby('date').ic.first(); print('h',h,'obs',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
print('turnover',signal.diff().abs().mean().mean())
signal.to_csv('scripts/miner_3_20301226_efficiency_breakout_signal.csv')
