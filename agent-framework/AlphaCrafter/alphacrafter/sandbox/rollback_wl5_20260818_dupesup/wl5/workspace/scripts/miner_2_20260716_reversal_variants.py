import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for a in A:
 try:
  x=get_stock_daily_data(a,days=2000); D[a]=x.set_index('date').close.astype(float)
 except: pass
p=pd.concat(D,axis=1).sort_index().ffill(); r=p.pct_change()
cs={'rev3':-(p/p.shift(3)-1),'rev5':-(p/p.shift(5)-1),'rev10':-(p/p.shift(10)-1),'volscaled_rev5':-(p/p.shift(5)-1)/(r.rolling(20).std()*np.sqrt(5)),'gap_reversal':-r.rolling(3).mean()}
for n,z in cs.items():
 ic=[]
 for i,dt in enumerate(p.index[:-1]):
  q=pd.concat([z.loc[dt],r.iloc[i+1]],axis=1).dropna()
  if len(q)>=8: ic.append(q.iloc[:,0].corr(q.iloc[:,1]))
 s=pd.Series(ic).dropna(); print(n,'n',len(s),'IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',(s>0).mean(),'turn',z.rank(pct=True).diff().abs().mean(axis=1).mean(),'cov',z.notna().sum(axis=1).mean()/15)
