import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); P[s]=d.sort_values('date').set_index('date').close.astype(float)
px=pd.DataFrame(P).sort_index().ffill(); px=px.loc[:'2030-05-01']; r=px.pct_change()
vol=r.rolling(40,min_periods=20).std()*np.sqrt(252)
cons=r.rolling(20,min_periods=15).mean()/r.abs().rolling(20,min_periods=15).mean()
f=(px.pct_change(20)/vol)*cons
f=f.sub(f.median(axis=1),axis=0)
for h in [5,10,20]:
 y=px.shift(-h)/px-1; a=[]; ns=[]; ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(dt)
 ic=pd.Series(a,index=ds).dropna()
 print('h',h,'dates',len(ic),'assets',len(P),'mean_n',round(np.mean(ns),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4),'coverage',round(len(ic)/len(px),4))
 for lo,hi in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2030-05-01')]:
  q=ic.loc[(ic.index>=lo)&(ic.index<=hi)]
  if len(q): print(' regime',lo,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
rank=f.rank(axis=1,pct=True)
print('turnover',round(float(rank.diff().abs().mean(axis=1).mean()),6),'mean valid coverage',round(float(f.notna().sum(axis=1).mean()/len(P)),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20300502_trend_consistency_signal.csv',index=False)
