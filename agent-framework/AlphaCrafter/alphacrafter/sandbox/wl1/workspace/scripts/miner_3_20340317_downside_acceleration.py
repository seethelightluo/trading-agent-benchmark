import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)==0: d=get_index_daily_data(s,5000)
 if d is not None and len(d): P[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
px=pd.DataFrame(P).sort_index().ffill(); r=px.pct_change()
# Candidate: downside-risk-adjusted trend acceleration. Short trend is compared with
# one-third long trend, then scaled by total volatility and penalized by downside share.
m15=np.log(px/px.shift(15)); m45=np.log(px/px.shift(45))
vol=r.rolling(30,min_periods=20).std()*np.sqrt(30)
down=np.sqrt((r.clip(upper=0)**2).rolling(30,min_periods=20).mean())*np.sqrt(30)
path=(r.rolling(30,min_periods=20).sum().abs()/(r.abs().rolling(30,min_periods=20).sum()+1e-12)).clip(0,1)
down_ratio=(down/(vol+1e-8)).clip(0,2)
f=((m15-m45/3)/(vol+1e-6))*(0.5+path)*(1-0.35*down_ratio)
f=f.shift(1); fr=px.pct_change(10).shift(-10)
ics=[];ns=[];ds=[]
for dt in f.index:
 a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(a)>=8:
  c=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
  if np.isfinite(c):ics.append(c);ns.append(len(a));ds.append(dt)
z=np.array(ics); rank=f.rank(axis=1,pct=True)
print('dates',len(z),'avgN %.2f'%np.mean(ns),'IC %.8f ICIR %.8f hit %.5f coverage %.5f turnover %.5f'%(z.mean(),z.mean()/z.std(ddof=1),np.mean(z>0),f.notna().sum(axis=1).mean()/len(U),rank.diff().abs().mean(axis=1).dropna().mean()))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 q=np.array([v for d,v in zip(ds,ics) if a<=str(d.year)<=b]); print('REG',a,b,len(q),'IC %.8f ICIR %.8f'%(q.mean(),q.mean()/q.std(ddof=1)))
f.loc[ds].to_csv('scripts/miner_3_20340317_downside_acceleration_signal.csv',index_label='date')
