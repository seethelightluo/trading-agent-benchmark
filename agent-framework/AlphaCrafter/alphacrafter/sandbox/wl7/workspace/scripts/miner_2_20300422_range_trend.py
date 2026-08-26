import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; a={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d)>120:
  d=d.copy();d.date=pd.to_datetime(d.date);a[s]=d.set_index('date').close.astype(float)
p=pd.concat(a,axis=1).sort_index().ffill();r=np.log(p).diff();v=r.rolling(20).std()
# medium-term trend location, volatility normalized and lagged
f=((p-p.rolling(120).min())/(p.rolling(120).max()-p.rolling(120).min())-.5)/v
f=f.shift(1)
for h in [5,10,20,40]:
 y=np.log(p.shift(-h)/p);q=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1]))
 q=pd.Series(q).dropna();print(h,len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean())
y=np.log(p.shift(-10)/p);z=[]
for dt in p.index:
 x=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(x)>=8:z.append((dt,x.iloc[:,0].corr(x.iloc[:,1]),len(x)))
q=pd.DataFrame(z,columns=['date','ic','n']).set_index('date')
for n,x in [('early',q.iloc[:len(q)//3]),('mid',q.iloc[len(q)//3:2*len(q)//3]),('late',q.iloc[2*len(q)//3:])]:print(n,len(x),x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1)*np.sqrt(len(x)),x.n.mean())
print('assets',len(a),'dates',len(p),'coverage',f.notna().sum(axis=1).div(len(a)).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
s=f.stack().rename('signal').reset_index();s.columns=['date','symbol','signal'];s.to_csv('scripts/miner_2_20300422_range_trend_signal.csv',index=False);q.reset_index().to_csv('scripts/miner_2_20300422_range_trend_ic.csv',index=False)
