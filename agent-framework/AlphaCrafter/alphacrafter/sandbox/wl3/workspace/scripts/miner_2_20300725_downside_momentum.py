import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300:d=get_index_daily_data(s,4000)
 if d is not None and len(d):D[s]=d.set_index('date')
px=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index().ffill();r=np.log(px).diff()
ret=r.rolling(10,min_periods=8).sum(); down=r.clip(upper=0).rolling(30,min_periods=20).std(); f=(ret/(down*np.sqrt(10)+1e-12)).shift(1)
for h in [1,3,5,10]:
 q=[]
 for i,dt in enumerate(px.index[:-h]):
  z=pd.concat([f.loc[dt],r.iloc[i+1:i+h+1].sum()],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q); print('H',h,'obs',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),np.mean(q>0)))
print('dates',len(px),'instruments',len(D),'coverage',f.notna().mean().mean())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20300725_downside_momentum_signal.csv',index=False)
