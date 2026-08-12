import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0:d=get_index_daily_data(s,5000)
 if d is not None and len(d): P[s]=d.drop_duplicates('date').set_index('date')['close'].astype(float)
px=pd.DataFrame(P).sort_index().ffill(); r=px.pct_change()
# Slow anchor-conditioned medium trend: 60d relative trend, normalized by 40d risk;
# defensive state is the average 40d anchor trend relative to cross-section median.
med40=r.rolling(40,min_periods=30).sum().median(axis=1)
trend=r.rolling(60,min_periods=45).sum(); risk=r.rolling(40,min_periods=30).std()
anchor=((trend['XAU']-trend.median(axis=1))+(trend['US10Y']-trend.median(axis=1))+(trend['CN10Y']-trend.median(axis=1)))/3
state=np.tanh(anchor/(risk.median(axis=1)+1e-6))
base=trend.sub(trend.median(axis=1),axis=0)/(risk+1e-6)
f=base.mul(1-0.25*state,axis=0) + base.mul(0.15*state,axis=0)
f=f.sub(f.median(axis=1),axis=0).shift(1)
rows=[]
for i in range(len(px)-20):
 for h in [1,5,10,20]:
  z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8: rows.append((px.index[i],h,len(z),z.x.corr(z.y,method='spearman')))
o=pd.DataFrame(rows,columns=['date','h','n','ic']); print('period',px.index.min(),px.index.max(),'assets',px.shape[1],'dates',o.date.nunique(),'avgN',o.n.mean())
for h in [1,5,10,20]:
 q=o[o.h==h].groupby('date').ic.first(); print('h',h,'obs',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
 for label,sub in [('2020-2025',q.loc[:'2025-12-31']),('2026+',q.loc['2026-01-01':]),('2029+',q.loc['2029-01-01':]),('2030+',q.loc['2030-01-01':])]:
  if len(sub)>20: print(' ',label,'n',len(sub),'IC %.6f ICIR %.6f'%(sub.mean(),sub.mean()/sub.std(ddof=1)))
print('coverage %.4f turnover %.6f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean().mean())); f.to_csv('scripts/miner_2_20310529_slow_anchor_signal.csv')
