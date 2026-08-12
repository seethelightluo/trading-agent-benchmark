import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 raw[s]=d.set_index(pd.to_datetime(d.date)).close
px=pd.DataFrame(raw).sort_index(); lr=np.log(px).diff()
ret60=np.log(px/px.shift(60)); med=ret60.median(axis=1)
vol=lr.rolling(60,min_periods=40).std()
f=(-(ret60.sub(med,axis=0))/(vol*np.sqrt(60)+1e-12)).shift(1)
ys={h:np.log(px.shift(-h)/px) for h in [1,5,10,20,40]}
for h,y in ys.items():
 vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c): vals.append((dt,c)); ns.append(len(z))
 q=pd.Series(dict(vals),dtype=float); q.index=pd.to_datetime(q.index)
 print('horizon',h,'dates',len(q),'avg_n',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6),'hit',round((q>0).mean(),4))
 for a,b in [(pd.Timestamp('2020-01-01'),pd.Timestamp('2022-12-31')),(pd.Timestamp('2023-01-01'),pd.Timestamp('2025-12-31')),(pd.Timestamp('2026-01-01'),pd.Timestamp('2028-12-31')),(pd.Timestamp('2029-01-01'),pd.Timestamp('2030-12-31')),(pd.Timestamp('2031-01-01'),pd.Timestamp('2032-02-05'))]:
  r=q[(q.index>=a)&(q.index<=b)]
  print('regime',a.date(),b.date(),'dates',len(r),'IC',round(r.mean(),6),'ICIR',round(r.mean()/r.std(),6))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20320205_residual_reversal_60d_signal.csv',index=False)
