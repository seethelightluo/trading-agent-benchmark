import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300:
  try: d=get_index_daily_data(s,4000)
  except FileNotFoundError: d=None
 if d is not None and len(d): D[s]=d.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().ffill(); r=np.log(px).diff(); m=r.mean(axis=1)
res=pd.DataFrame(index=r.index,columns=r.columns,dtype=float)
for s in r:
 cov=r[s].rolling(60,min_periods=40).cov(m); vv=m.rolling(60,min_periods=40).var()
 res[s]=r[s]-cov/(vv+1e-12)*m
vol=res.rolling(40,min_periods=25).std()
for look in [3,5,10]:
 f=(-(res.rolling(look,min_periods=look).sum())/(vol*np.sqrt(look)+1e-12)).shift(1)
 print('LOOK',look)
 for h in [1,3,5,10]:
  q=[]
  for i,dt in enumerate(px.index[:-h]):
   z=pd.concat([f.loc[dt],np.log(px.iloc[i+h]/px.iloc[i])],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
  q=pd.DataFrame(q,columns=['date','n','ic']).set_index('date')
  print('H',h,'obs',len(q),'avgN %.2f IC %.6f ICIR %.6f hit %.4f'%(q.n.mean(),q.ic.mean(),q.ic.mean()/(q.ic.std(ddof=1)+1e-12),(q.ic>0).mean()))
 print('coverage %.4f dates %d instruments %d'%(f.notna().mean().mean(),len(px),len(D)))
 if look==5:
  f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20300919_residual_meanrev_signal.csv',index=False)
  print('artifact written')
