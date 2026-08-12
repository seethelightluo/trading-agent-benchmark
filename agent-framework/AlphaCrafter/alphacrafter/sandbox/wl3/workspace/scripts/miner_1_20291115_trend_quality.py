import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300: d=get_index_daily_data(s,4000)
 if d is not None and len(d): C[s]=d.set_index('date').close.astype(float)
px=pd.DataFrame(C).sort_index(); lr=np.log(px).diff()
# Trend quality: medium-term return penalized by downside path volatility; lagged.
ret=np.log(px/px.shift(60))
down=lr.clip(upper=0).pow(2).rolling(60).sum().pow(.5)
f=(ret/(down+1e-8)).shift(1)
f=f.sub(f.median(axis=1),axis=0)
for h in [1,3,5,10]:
 rows=[]
 for i,dt in enumerate(px.index):
  if i+h>=len(px): break
  z=pd.concat([f.loc[dt],np.log(px.iloc[i+h]/px.iloc[i])],axis=1).dropna()
  if len(z)>=8: rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
 x=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
 print('H',h,'obs',len(x),'avgN %.2f'%x.n.mean(),'IC %.6f ICIR %.6f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean()))
 for lab,z in [('2020_22',x.loc['2020-01-01':'2022-12-31']),('2023_25',x.loc['2023-01-01':'2025-12-31']),('2026_27',x.loc['2026-01-01':'2027-12-31']),('2028_29',x.loc['2028-01-01':]),('recent250',x.tail(250))]:
  print(lab,len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.ic.mean(),z.ic.mean()/z.ic.std(),(z.ic>0).mean()))
rr=f.rank(axis=1,pct=True); print('dates',len(px),'instruments',len(C),'coverage %.4f'%(f.notna().sum().sum()/(len(f)*len(C))),'turnover %.4f'%(rr.diff().abs().mean(axis=1).mean()))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20291115_trend_quality_signal.csv',index=False)
