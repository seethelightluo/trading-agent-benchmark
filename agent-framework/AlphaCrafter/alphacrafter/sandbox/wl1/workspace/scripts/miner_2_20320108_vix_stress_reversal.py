import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-01-07'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000); d['date']=pd.to_datetime(d.date); raw[s]=d[d.date<=cut].set_index('date').close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); vol=r.rolling(30,min_periods=20).std()
v=pd.read_csv('../persistent/index_data/VIX.csv'); v['date']=pd.to_datetime(v.date); v=v[v.date<=cut].set_index('date').close.reindex(px.index).ffill()
shock=np.log(v).diff(5); med=shock.rolling(252,min_periods=60).median(); scale=(shock-med).clip(lower=0).rolling(20,min_periods=5).mean()/(shock.rolling(252,min_periods=60).std()+1e-12)
base=-np.log(px/px.shift(5))/(vol*np.sqrt(5)+1e-12)
f=base.mul(1+scale,axis=0).shift(1)
fw={h:np.log(px.shift(-h)/px) for h in [1,5,10,20]}; qs={}
for h,y in fw.items():
 vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))); ns.append(len(z))
 q=pd.Series([x[1] for x in vals],index=pd.to_datetime([x[0] for x in vals])).dropna(); qs[h]=q
 print('horizon',h,'dates',len(q),'avg_n',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6),'hit',round((q>0).mean(),4))
for a,b in [(pd.Timestamp('2020-01-01'),pd.Timestamp('2022-12-31')),(pd.Timestamp('2023-01-01'),pd.Timestamp('2025-12-31')),(pd.Timestamp('2026-01-01'),pd.Timestamp('2028-12-31')),(pd.Timestamp('2029-01-01'),cut)]:
 q=qs[10][(qs[10].index>=a)&(qs[10].index<=b)]; print('regime',a.date(),b.date(),'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20320108_vix_stress_reversal_signal.csv',index=False)
print('signal_rows',len(out),'px_dates',len(px),'v_nonnull',v.notna().sum())
