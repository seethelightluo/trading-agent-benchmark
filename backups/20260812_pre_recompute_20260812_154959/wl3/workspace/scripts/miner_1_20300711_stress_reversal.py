import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300: d=get_index_daily_data(s,4000)
 D[s]=d.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().ffill(); r=np.log(px).diff()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(px.index).ffill()
base=-r.rolling(5,min_periods=5).sum(); base=base.sub(base.median(axis=1),axis=0)
vz=vix/vix.rolling(120,min_periods=60).median()-1
active=(vz>0.10).rolling(3,min_periods=3).sum().eq(3)
scaled=base.mul(1+vz.clip(0,1.5),axis=0)
f=scaled.mask(~active,axis=0).shift(1)
for h in [1,3,5,10]:
 q=[]
 for i,dt in enumerate(px.index[:-h]):
  z=pd.concat([f.loc[dt],np.log(px.iloc[i+h]/px.iloc[i])],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q); print('H',h,'obs',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/(q.std(ddof=1)+1e-12),(q>0).mean()) if len(q)>1 else 'insufficient')
print('dates',len(px),'instruments',len(D),'active_dates',active.sum(),'coverage_active',f.notna().sum().sum()/max(1,active.shift(1).sum()*len(U)))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20300711_stress_reversal_signal.csv',index=False)
