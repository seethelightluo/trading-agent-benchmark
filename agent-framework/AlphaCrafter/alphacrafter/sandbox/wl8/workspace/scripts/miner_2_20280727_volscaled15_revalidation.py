import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d): D[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').sort_index().close.astype(float)
px=pd.concat(D,axis=1).sort_index(); r=px.pct_change(); f=-(px.shift(1)/px.shift(16)-1)/(r.shift(1).rolling(20,min_periods=15).std()*np.sqrt(20))
rows=[]
for i,dt in enumerate(px.index):
 for h in [1,3,5,10]:
  if i+h>=len(px): continue
  z=pd.concat([f.loc[dt],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
o=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,3,5,10]:
 z=o[o.h==h]; q=z.ic.dropna(); print('h',h,'dates',len(q),'avgN',z.n.mean(),'IC %.6f ICIR %.6f hit %.4f coverage %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),f.notna().sum().sum()/f.size))
 for y in [2026,2027,2028]:
  a=z[z.date.dt.year==y].ic.dropna(); print(y,len(a),'IC %.6f ICIR %.6f'%(a.mean(),a.mean()/a.std(ddof=1)) if len(a)>1 else '')
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20280727_volscaled15_revalidation_signal.csv',index=False)
