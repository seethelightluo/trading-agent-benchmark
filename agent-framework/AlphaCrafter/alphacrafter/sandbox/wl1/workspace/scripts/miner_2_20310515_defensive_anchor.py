import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0:d=get_index_daily_data(s,5000)
 if d is not None and len(d):P[s]=d.drop_duplicates('date').set_index('date')['close'].astype(float)
px=pd.DataFrame(P).sort_index().ffill(); r=px.pct_change(); v20=r.rolling(20,min_periods=15).std(); mom=r.rolling(20,min_periods=15).sum(); short=r.rolling(5,min_periods=4).sum(); cs=mom.copy(); med=cs.median(axis=1)
defstate=((cs['XAU']-med)+(cs['US10Y']-med)+(cs['CN10Y']-med))/3
state=np.tanh(defstate/(v20.median(axis=1)+1e-6))
rel=mom.sub(med,axis=0)/(v20+1e-6); relcs=cs.sub(med,axis=0)/(v20+1e-6)
f=rel.mul(1-0.30*state,axis=0)+relcs.mul(0.20*state,axis=0)-short.div(v20+1e-6).mul(0.20)
f=f.sub(f.median(axis=1),axis=0).shift(1); rows=[]
for i in range(len(px)-20):
 for h in [1,5,10,20]:
  z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8:rows.append((px.index[i],h,len(z),z.x.corr(z.y,method='spearman')))
o=pd.DataFrame(rows,columns=['date','h','n','ic']);print('period',px.index.min(),px.index.max(),'assets',px.shape[1],'dates',o.date.nunique(),'avgN',o.n.mean())
for h in [1,5,10,20]:
 q=o[o.h==h].groupby('date').ic.first();print('h',h,'obs',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
 for label,sub in [('2020-2025',q.loc[:'2025-12-31']),('2026+',q.loc['2026-01-01':]),('2029+',q.loc['2029-01-01':]),('2030+',q.loc['2030-01-01':])]:
  if len(sub)>20: print(' ',label,'n',len(sub),'IC %.6f ICIR %.6f'%(sub.mean(),sub.mean()/sub.std(ddof=1)))
print('coverage %.4f turnover %.6f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean().mean()));f.to_csv('scripts/miner_2_20310515_defensive_anchor_signal.csv')
