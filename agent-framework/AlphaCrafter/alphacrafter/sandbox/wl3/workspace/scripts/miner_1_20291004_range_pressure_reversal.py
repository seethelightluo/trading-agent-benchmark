import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}; O={}; H={}; L={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300: d=get_index_daily_data(s,4000)
 if d is not None and len(d):
  q=d.set_index('date'); C[s]=q.close.astype(float); O[s]=q.open.astype(float); H[s]=q.high.astype(float); L[s]=q.low.astype(float)
px=pd.DataFrame(C).sort_index(); op=pd.DataFrame(O).reindex(px.index); hi=pd.DataFrame(H).reindex(px.index); lo=pd.DataFrame(L).reindex(px.index)
r=np.log(px).diff(); vol30=r.rolling(30).std();
# Range-pressure reversal: reverse recent signed candle pressure, scaled by the 3d close shock;
# range normalization avoids price-level effects, while a trend multiplier suppresses fading strong trends.
rng=(hi-lo).replace(0,np.nan); pressure=((px-op)/rng).clip(-3,3)
shock=np.log(px/px.shift(3))/(vol30*np.sqrt(3))
trend=np.log(px/px.shift(30)); gate=1/(1+np.exp(trend/0.10)) # more reversal weight after negative/flat trend
f=(-shock*(1+pressure.rolling(5).mean().clip(-1,1))*gate).shift(1)
f=f.sub(f.median(axis=1),axis=0)
for h in [1,3,5,10]:
 rows=[]
 for i,dt in enumerate(px.index):
  if i+h>=len(px): break
  z=pd.concat([f.loc[dt],np.log(px.iloc[i+h]/px.iloc[i])],axis=1).dropna()
  if len(z)>=8: rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
 x=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
 print('H',h,'obs',len(x),'avgN %.2f'%x.n.mean(),'IC %.6f ICIR %.6f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean()))
 for lab,z in [('2026_29',x.loc['2026-01-01':]),('recent250',x.tail(250))]: print(lab,len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.ic.mean(),z.ic.mean()/z.ic.std(),(z.ic>0).mean()))
rr=f.rank(axis=1,pct=True); print('dates',len(px),'instruments',len(C),'coverage %.4f'%(f.notna().sum().sum()/(len(f)*len(C))),'turnover %.4f'%rr.diff().abs().mean(axis=1).mean())
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20291004_range_pressure_reversal_signal.csv',index=False)
