import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000); d['date']=pd.to_datetime(d.date); raw[s]=d.set_index('date').close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); v=r.rolling(30).std(); r5=np.log(px/px.shift(5)); med=r5.median(axis=1); disp=r5.sub(med,axis=0).abs().median(axis=1); th=disp.rolling(120,min_periods=60).median(); active=(disp>th).astype(float); f=(-(r5.sub(med,axis=0))/(v*np.sqrt(5)+1e-12)).mul(active,axis=0).shift(1); ys={h:np.log(px.shift(-h)/px) for h in [1,5,10,20]}; qs={}
for h,y in ys.items():
 vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c): vals.append((dt,c)); ns.append(len(z))
 q=pd.Series(dict(vals),dtype=float); q.index=pd.to_datetime(q.index); qs[h]=q
 print('horizon',h,'dates',len(q),'avg_n',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6),'hit',round((q>0).mean(),4))
for a,b in [(pd.Timestamp('2020-01-01'),pd.Timestamp('2022-12-31')),(pd.Timestamp('2023-01-01'),pd.Timestamp('2025-12-31')),(pd.Timestamp('2026-01-01'),pd.Timestamp('2028-12-31')),(pd.Timestamp('2029-01-01'),pd.Timestamp('2030-12-31')),(pd.Timestamp('2031-01-01'),pd.Timestamp('2032-01-07'))]:
 q=qs[10][(qs[10].index>=a)&(qs[10].index<=b)]; print('regime',a.date(),b.date(),'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20320122_dispersion_relative_reversal_signal.csv',index=False)
