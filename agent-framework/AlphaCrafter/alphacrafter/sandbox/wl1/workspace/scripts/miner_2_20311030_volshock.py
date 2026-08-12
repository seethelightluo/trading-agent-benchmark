import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=get_stock_daily_data(s,5000)
 if x is None or len(x)==0:x=get_index_daily_data(s,5000)
 if x is not None and len(x)>100:
  x=x[['date','close']].copy();x.date=pd.to_datetime(x.date);D[s]=x.drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=np.log(p).diff(); v=r.rolling(20,min_periods=10).std(); cross=v.div(v.median(axis=1),axis=0)
# Contrarian shock, increasingly active when instrument volatility is unusually high; lagged.
f=(-np.log(p/p.shift(5))/(v+1e-8))*(0.75+0.5*cross.clip(0.5,2.0));f=f.shift(1)
rows=[]
for h in [1,5,10,20]:
 a=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i],np.log(p.iloc[i+h]/p.iloc[i])],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(a).dropna();rows.append((h,len(q),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
print('period',p.index.min().date(),p.index.max().date(),'dates',len(p),'assets',len(D),'observations',rows)
print('coverage',round(f.notna().mean().mean(),5),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
for a,b in [(0,int(len(p)*.33)),(int(len(p)*.33),int(len(p)*.66)),(int(len(p)*.66),len(p))]:
 q=[]
 for i in range(a,min(b,len(p)-10)):
  z=pd.concat([f.iloc[i],np.log(p.iloc[i+10]/p.iloc[i])],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(q).dropna();print('regime',p.index[a].date(),p.index[min(b-1,len(p)-1)].date(),len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20311030_volshock_signal.csv',index=False)
