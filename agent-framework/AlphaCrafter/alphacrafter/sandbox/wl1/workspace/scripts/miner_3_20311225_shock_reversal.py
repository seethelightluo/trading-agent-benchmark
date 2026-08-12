import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2031-12-24'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000); d['date']=pd.to_datetime(d.date); d=d[d.date<=cut].sort_values('date'); raw[s]=d.set_index('date').close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); rv=r.rolling(20).std(); shock=np.log(px/px.shift(5))/(rv*np.sqrt(5)+1e-12)
# contrarian response to statistically large five-day shocks; neutral when shock is small
f=(-shock*(shock.abs()>1.0)).shift(1); fw={h:np.log(px.shift(-h)/px) for h in [1,5,10,20]}
allq={}
for h,y in fw.items():
 vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))); ns.append(len(z))
 q=pd.Series(dict(vals)).dropna(); allq[h]=q
 print('horizon',h,'dates',len(q),'avg_n',round(float(np.mean(ns)),2),'coverage',round(float(np.mean(ns)/15),4),'IC',round(float(q.mean()),6),'ICIR',round(float(q.mean()/q.std()),6),'hit',round(float((q>0).mean()),4))
for a,b in [(pd.Timestamp('2020-01-01'),pd.Timestamp('2022-12-31')),(pd.Timestamp('2023-01-01'),pd.Timestamp('2025-12-31')),(pd.Timestamp('2026-01-01'),pd.Timestamp('2028-12-31')),(pd.Timestamp('2029-01-01'),pd.Timestamp('2030-12-31')),(pd.Timestamp('2031-01-01'),cut)]:
 q=allq[10][(allq[10].index>=a)&(allq[10].index<=b)]
 print('regime',a.date(),b.date(),'dates',len(q),'IC',round(float(q.mean()),6),'ICIR',round(float(q.mean()/q.std()),6) if len(q)>1 else np.nan)
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20311225_shock_reversal_signal.csv',index=False)
