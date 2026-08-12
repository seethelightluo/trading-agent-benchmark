import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x)>80:
  x=x[['date','close']]; x.date=pd.to_datetime(x.date); D[s]=x.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff(); v=r.rolling(20).std()
# contrarian short-horizon reversal, damped when cross-sectional dispersion is high
cs=r.sub(r.mean(axis=1),axis=0); disp=cs.rolling(10).std().mean(axis=1); damp=1/(1+disp/disp.rolling(120).median())
f=(-np.log(p/p.shift(3))/(v+1e-8)).mul(damp,axis=0)
for h in [1,5,10,20]:
 q=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i],np.log(p.iloc[i+h]/p.iloc[i])],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(q).dropna();print(h,len(q),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean())
print('coverage',f.notna().sum(axis=1).mean()/15,'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20311016_short_reversal_signal.csv',index=False)
