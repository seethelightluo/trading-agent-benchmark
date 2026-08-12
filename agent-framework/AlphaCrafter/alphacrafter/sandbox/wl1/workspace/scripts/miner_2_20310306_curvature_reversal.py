import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0:d=get_index_daily_data(s,5000)
 if d is not None and len(d): px[s]=d.drop_duplicates('date').set_index('date')['close'].astype(float)
p=pd.DataFrame(px).sort_index().ffill(); r=p.pct_change()
# Trend-curvature with short-term reversal: intermediate 20d trend relative to
# long 60d baseline, penalized by the latest 5d move; risk normalized and lagged.
mid=r.rolling(20,min_periods=15).sum(); long=r.rolling(60,min_periods=40).sum()/3
vol=r.rolling(40,min_periods=25).std(); short=r.rolling(5,min_periods=4).sum()
f=((mid-long)-0.35*short)/(vol+1e-6)
f=f.sub(f.median(axis=1),axis=0).shift(1)
rows=[]
for i,dt in enumerate(p.index[:-20]):
 for h in [1,5,10,20]:
  z=pd.concat([f.iloc[i].rename('x'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,len(z),z.x.corr(z.y,method='spearman')))
o=pd.DataFrame(rows,columns=['date','h','n','ic']); print('period',p.index.min(),p.index.max(),'assets',p.shape[1],'dates',o.date.nunique(),'avgN',o.n.mean())
for h in [1,5,10,20]:
 q=o[o.h==h].groupby('date').ic.first(); print('h',h,'obs',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
 for label,sub in [('2020-2025',q.loc[:'2025-12-31']),('2026+',q.loc['2026-01-01':]),('2029+',q.loc['2029-01-01':]),('2030+',q.loc['2030-01-01':])]:
  if len(sub)>20: print(' ',label,'n',len(sub),'IC %.6f ICIR %.6f'%(sub.mean(),sub.mean()/sub.std(ddof=1)))
print('coverage %.4f turnover %.6f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean().mean())); f.to_csv('scripts/miner_2_20310306_curvature_reversal_signal.csv')
