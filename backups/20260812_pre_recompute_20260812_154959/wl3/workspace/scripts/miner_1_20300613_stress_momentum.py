import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300:d=get_index_daily_data(s,4000)
 if d is not None and len(d):D[s]=d.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().ffill(); r=np.log(px).diff(); vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.astype(float).reindex(px.index).ffill()
# Defensive cross-sectional relative strength: 20d residual momentum, activated in persistent high-vol regime, lagged.
raw=r.rolling(20).sum(); f=raw.sub(raw.median(axis=1),axis=0).where(vix>vix.rolling(60).median(),0).shift(1)
for h in [1,3,5,10]:
 q=[]
 for i,dt in enumerate(px.index[:-h]):
  z=pd.concat([f.loc[dt],np.log(px.iloc[i+h]/px.iloc[i])],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 q=pd.DataFrame(q,columns=['date','ic']).set_index('date'); print('H',h,'obs',len(q),'N',len(D),'IC %.6f ICIR %.6f hit %.4f'%(q.ic.mean(),q.ic.mean()/(q.ic.std(ddof=1)+1e-12),(q.ic>0).mean()))
 for a,b in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030')]:
  y=q.loc[a:b].ic
  if len(y):print(a+'-'+b,len(y),'IC %.6f ICIR %.6f'%(y.mean(),y.mean()/(y.std(ddof=1)+1e-12)))
print('dates',len(px),'coverage %.4f'%f.notna().mean().mean(),'active',int((vix>vix.rolling(60).median()).sum()))
f.stack().rename('signal').reset_index().to_csv('scripts/miner_1_20300613_stress_momentum_signal.csv',index=False)
