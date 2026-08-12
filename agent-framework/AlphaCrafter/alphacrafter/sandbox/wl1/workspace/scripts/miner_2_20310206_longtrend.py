import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0:d=get_index_daily_data(s,5000)
 if d is not None and len(d):px[s]=d.drop_duplicates('date').set_index('date')['close'].astype(float)
p=pd.DataFrame(px).sort_index().ffill();r=p.pct_change(); mom=r.rolling(40,min_periods=25).sum(); med=mom.median(axis=1); vol=r.rolling(60,min_periods=40).std(); persist=r.gt(0).rolling(40,min_periods=25).mean(); f=(mom.sub(med,axis=0)/(vol+1e-6)*(0.5+persist)).shift(1)
rows=[]
for i,dt in enumerate(p.index[:-20]):
 for h in [1,5,10,20]:
  z=pd.concat([f.iloc[i].rename('x'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8:rows.append((dt,h,len(z),z.x.corr(z.y,method='spearman')))
o=pd.DataFrame(rows,columns=['date','h','n','ic']);print('period',p.index.min(),p.index.max(),'assets',p.shape[1],'dates',o.date.nunique(),'avgN',o.n.mean())
for h in [1,5,10,20]:
 q=o[o.h==h].groupby('date').ic.first();print('h',h,'obs',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
print('coverage %.4f turnover %.6f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean().mean()));f.to_csv('scripts/miner_2_20310206_longtrend_signal.csv')
