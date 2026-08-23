import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 d=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:d=fn(s,days=4000)
  except:pass
  if d is not None and len(d)>300:break
 if d is not None:D[s]=pd.Series(d.close.values,index=pd.to_datetime(d.date))
P=pd.DataFrame(D).sort_index();R=P.pct_change(); mom=P.pct_change(10); vol=R.rolling(20).std()*np.sqrt(252); br=R.gt(0).rolling(10).mean();F=mom/vol*((br-.5)*2)
for h in [10]:
 x=[]
 for t in P.index:
  z=pd.concat([F.loc[t],(P.shift(-h)/P-1).loc[t]],axis=1).dropna()
  if len(z)>=8:x.append(z.iloc[:,0].corr(z.iloc[:,1]))
 a=pd.Series(x).dropna();print('assets',len(D),'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(),'hit',(a>0).mean(),'turn',F.rank(pct=True).diff().abs().mean().mean(),'coverage',F.notna().mean().mean())
F.index.name='date';F.to_csv('scripts/miner_2_20340914_fast_persistence_signal.csv')
