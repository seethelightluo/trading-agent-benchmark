import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=get_stock_daily_data(s,4000)
 if x is None:x=get_index_daily_data(s,4000)
 if x is not None:D[s]=pd.Series(x.close.astype(float).values,index=pd.to_datetime(x.date))
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(fill_method=None); sig=(-p.pct_change(5,fill_method=None)/r.rolling(20).std()).shift(1)
print('assets',len(D),'rows',len(p),'valid',sig.notna().sum(axis=1).describe())
for h in [5,10,20,40]:
 f=p.shift(-h)/p-1;out=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:out.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(out,columns=['date','ic','n']).set_index('date');print('h',h,'dates',len(q),'avgN',q.n.mean(),'coverage',q.n.mean()/15,'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(),'hit',(q.ic>0).mean())
 if h==10:
  print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean());q.to_csv('scripts/miner_1_20340915_shock_reversal_signal.csv')
