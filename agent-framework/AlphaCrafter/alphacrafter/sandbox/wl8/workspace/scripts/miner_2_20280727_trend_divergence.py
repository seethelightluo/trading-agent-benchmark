import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d):
  d=d.copy(); d['date']=pd.to_datetime(d.date); D[s]=d.set_index('date').sort_index().close.astype(float)
px=pd.concat(D,axis=1).sort_index(); r=px.pct_change()
# lagged inputs; trend divergence = 60d trend minus recent 5d move
f=(px.shift(1)/px.shift(61)-1 - (px.shift(1)/px.shift(6)-1))
f=f.replace([np.inf,-np.inf],np.nan)
rows=[]
for i,dt in enumerate(px.index):
 for h in [1,3,5,10]:
  if i+h>=len(px): continue
  z=pd.concat([f.loc[dt],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
o=pd.DataFrame(rows,columns=['date','h','ic','n'])
print('assets',px.shape[1],'dates',px.index.min(),px.index.max(),'rows',len(o))
for h in [1,3,5,10]:
 z=o[o.h==h]; q=z.ic.dropna()
 print(h,'dates',len(q),'avgN %.2f IC %.6f ICIR %.6f hit %.3f'%(z.n.mean(),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
 for lab,mask in [('2020-22',z.date.dt.year<=2022),('2023-25',z.date.dt.year.between(2023,2025)),('2026',z.date.dt.year==2026),('2027',z.date.dt.year==2027),('2028',z.date.dt.year==2028),('recent280',z.date>=z.date.max()-pd.Timedelta(days=280))]:
  q=z[mask].ic.dropna()
  if len(q): print(' ',lab,len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
s=f.stack().rename('signal').reset_index(); s.columns=['date','symbol','signal']; s.to_csv('scripts/miner_2_20280727_trend_divergence_signal.csv',index=False)
