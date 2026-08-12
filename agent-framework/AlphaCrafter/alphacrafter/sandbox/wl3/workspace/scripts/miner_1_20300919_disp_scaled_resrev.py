import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is not None and len(d): D[s]=d.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().ffill(); r=np.log(px).diff(); m=r.mean(axis=1)
res=pd.DataFrame(index=r.index,columns=r.columns,dtype=float)
for s in r:
 beta=r[s].rolling(60,min_periods=40).cov(m)/(m.rolling(60,min_periods=40).var()+1e-12); res[s]=r[s]-beta*m
vol=res.rolling(40,min_periods=25).std(); disp=res.rolling(1).std(axis=1)
f=(-(res.rolling(3,min_periods=3).sum())/(vol*np.sqrt(3)+1e-12)).mul(disp/disp.rolling(60,min_periods=30).median(),axis=0).shift(1)
for h in [1,3,5,10]:
 q=[]
 for i,dt in enumerate(px.index[:-h]):
  z=pd.concat([f.loc[dt],np.log(px.iloc[i+h]/px.iloc[i])],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(q); print('H',h,'obs',len(a),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
print('dates',len(px),'instruments',len(D),'coverage',f.notna().mean().mean())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20300919_disp_scaled_resrev_signal.csv',index=False)
