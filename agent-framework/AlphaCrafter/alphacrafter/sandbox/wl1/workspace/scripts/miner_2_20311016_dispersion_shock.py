import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,5000)
 if x is None or len(x)==0: x=get_index_daily_data(s,5000)
 if x is not None and len(x)>100:
  x=x[['date','close']].copy(); x.date=pd.to_datetime(x.date); D[s]=x.drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff()
mom=np.log(p/p.shift(30)); short=np.log(p/p.shift(5)); vol=r.rolling(40,min_periods=20).std(); disp=r.sub(r.mean(axis=1),axis=0).rolling(20,min_periods=10).std()
f=(mom-0.8*short)/(vol+1e-8)/(1+disp/disp.rolling(120,min_periods=60).median())
f=f.shift(1)
rows=[]
for h in [1,5,10,20]:
 vals=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i],np.log(p.iloc[i+h]/p.iloc[i])],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(vals).dropna(); rows.append((h,len(q),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
print('dates',len(p),'assets',len(D),'observations',rows)
rank=f.rank(axis=1,pct=True); print('coverage',f.notna().mean().mean(),'turnover',rank.diff().abs().mean().mean())
for a,b in [(0,int(len(p)*.33)),(int(len(p)*.33),int(len(p)*.66)),(int(len(p)*.66),len(p))]:
 vals=[]
 for i in range(a,min(b,len(p)-20)):
  z=pd.concat([f.iloc[i],np.log(p.iloc[i+20]/p.iloc[i])],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(vals).dropna(); print('regime',p.index[a],p.index[min(b-1,len(p)-1)],len(q),q.mean(),q.mean()/q.std(ddof=1))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20311016_dispersion_shock_signal.csv',index=False)
