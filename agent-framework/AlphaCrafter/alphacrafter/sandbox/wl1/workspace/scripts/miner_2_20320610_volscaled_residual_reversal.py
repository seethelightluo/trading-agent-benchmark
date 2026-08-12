import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-06-09'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); d=d[d.date<=cut].set_index('date').sort_index(); raw[s]=d.close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); b=r.mean(axis=1); e=r.sub(b,axis=0)
rv10=e.rolling(10,min_periods=8).std(); rv60=e.rolling(60,min_periods=40).std()
# Contrarian residual trend, emphasizing assets whose recent residual risk is elevated
f=-e.rolling(20,min_periods=15).sum()*(rv10/rv60).clip(.5,2.5)
f=f.where((e.abs().rolling(5,min_periods=5).max()<rv60*3).fillna(False)).shift(1)
y=np.log(px.shift(-10)/px); vals=[]; ns=[]; dates=[]; turns=[]
for i,dt in enumerate(f.index):
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); dates.append(dt)
 if i:
  q=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
  if len(q)>=8: turns.append(q.iloc[:,0].rank().sub(q.iloc[:,1].rank()).abs().mean()/len(q))
s=pd.Series(vals,index=pd.DatetimeIndex(dates)); print('assets',len(raw),'dates',len(px),'valid',len(s),'avgN',np.mean(ns),'coverage',len(s)/len(px),'IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',np.mean(s>0),'turn',np.mean(turns))
for a,b in [('2024','2026'),('2027','2029'),('2030','2032')]:
 q=s[(s.index.year>=int(a))&(s.index.year<=int(b))]; print(a,b,len(q),q.mean(),q.mean()/q.std())
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20320610_volscaled_residual_reversal_signal.csv',index=False)
