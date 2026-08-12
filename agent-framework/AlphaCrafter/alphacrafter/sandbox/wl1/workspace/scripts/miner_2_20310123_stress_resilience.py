import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0: d=get_index_daily_data(s,5000)
 if d is not None and len(d): px[s]=d.drop_duplicates('date').set_index('date')['close'].astype(float)
p=pd.DataFrame(px).sort_index().ffill(); r=p.pct_change(); spx=r['SPX']; beta=r.rolling(60,min_periods=40).cov(spx).div(spx.rolling(60,min_periods=40).var()+1e-10,axis=0); resid=r-beta.mul(spx,axis=0)
vix=pd.read_csv('../persistent/index_data/VIX.csv'); vix['date']=pd.to_datetime(vix['date']); vv=vix.set_index('date')['close'].astype(float).reindex(p.index).ffill(); stress=(vv>vv.rolling(252,min_periods=126).quantile(.65)).astype(float)
f=(resid.rolling(20,min_periods=15).sum().mul(stress,axis=0)+r.rolling(20,min_periods=15).sum().mul(1-stress,axis=0)).shift(1)
rows=[]
for i,dt in enumerate(p.index[:-20]):
 for h in [1,5,10,20]:
  z=pd.concat([f.loc[dt].rename('x'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,len(z),z.x.corr(z.y,method='spearman')))
o=pd.DataFrame(rows,columns=['date','h','n','ic']); print('period',p.index.min(),p.index.max(),'assets',p.shape[1],'dates',o.date.nunique(),'avgN',o.n.mean())
for h in [1,5,10,20]:
 q=o[o.h==h].groupby('date').ic.first(); print('h',h,'obs',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
 for label,sub in [('2020-2025',q.loc[:'2025-12-31']),('2026+',q.loc['2026-01-01':]),('2029+',q.loc['2029-01-01':]),('2030+',q.loc['2030-01-01':])]:
  if len(sub)>20: print(' ',label,'n',len(sub),'IC %.6f ICIR %.6f'%(sub.mean(),sub.mean()/sub.std(ddof=1)))
print('coverage %.4f turnover %.6f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean().mean())); f.to_csv('scripts/miner_2_20310123_stress_resilience_signal.csv')
