import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={};O={};H={};L={};V={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0:d=get_index_daily_data(s,5000)
 d=d.drop_duplicates('date');ix=pd.to_datetime(d.date)
 for x,c in [(C,'close'),(O,'open'),(H,'high'),(L,'low'),(V,'volume')]:x[s]=d.set_index(ix)[c]
p=pd.DataFrame(C).sort_index().ffill(); o=pd.DataFrame(O).reindex(p.index).ffill(); h=pd.DataFrame(H).reindex(p.index).ffill(); l=pd.DataFrame(L).reindex(p.index).ffill()
# Reversal of abnormal close location: a lagged, range-normalized daily shock, smoothed over 3 sessions.
rng=(h-l).replace(0,np.nan); shock=((p-o)/rng).clip(-3,3)
# large directional close-location shocks tend to mean revert; normalize by recent range and smooth
signal=(-shock.rolling(3,min_periods=2).mean()).rank(axis=1,pct=True).shift(1)
rows=[]
for i in range(len(p)-21):
 for z in [1,5,10,20]:
  q=pd.concat([signal.iloc[i].rename('x'),(p.iloc[i+z]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8:rows.append((p.index[i],z,q.x.corr(q.y,method='spearman'),len(q)))
a=pd.DataFrame(rows,columns=['date','h','ic','n']);print('dates',a.date.nunique(),'assets',p.shape[1],'avgN',a.groupby('date').n.first().mean(),'coverage',signal.notna().mean().mean())
for z in [1,5,10,20]:
 q=a[a.h==z].groupby('date').ic.first();print('h',z,'obs',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
print('turnover',signal.diff().abs().mean().mean());signal.to_csv('scripts/miner_3_20301226_range_reversal_signal.csv')
