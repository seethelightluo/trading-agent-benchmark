import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 d=get_stock_daily_data(s,4500); return d if d is not None and len(d)>100 else get_index_daily_data(s,4500)
p=pd.concat({s:g(s).set_index('date').close for s in U},axis=1).sort_index().ffill(); r=p.pct_change(); neg=r.where(r<0); dd=np.sqrt((neg**2).rolling(30).mean()); f=(r.rolling(15).sum()/(dd*np.sqrt(15)+1e-6)).shift(1)
D={h:[] for h in [1,3,5,10]}; dates={h:[] for h in D}
for dt in f.index:
 i=p.index.get_loc(dt)
 for h in D:
  if i+h>=len(p): continue
  z=pd.concat([f.loc[dt],p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: D[h].append(z.iloc[:,0].corr(z.iloc[:,1])); dates[h].append(dt)
q=pd.Series(D[1],index=dates[1]); print('universe',len(p.columns),'dates',len(p),'IC dates',len(q),'avg n 15')
print('IC %.6f ICIR %.6f hit %.4f coverage %.4f turnover %.4f'%(q.mean(),q.mean()/q.std(),(q>0).mean(),f.notna().sum().sum()/(f.shape[0]*15),(f.diff().abs()>0.15).sum().sum()/f.notna().sum().sum()))
for a,b in [('2020','2022'),('2023','2025'),('2026','2031'),('2032','2032')]:
 z=q.loc[a:b]
 if len(z): print(a,b,len(z),'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std()))
for h,v in D.items(): print('decay',h,np.nanmean(v),len(v))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20320318_downside_adjusted_momentum_signal.csv',index=False)
