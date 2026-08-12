import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000); raw[s]=d.set_index(pd.to_datetime(d.date)).close
px=pd.DataFrame(raw).sort_index(); lr=np.log(px).diff(); ret40=np.log(px/px.shift(40)); med=ret40.median(axis=1); vol=lr.rolling(40,min_periods=30).std();
# contrarian relative 40d return, volatility normalized, lagged one session
f=(-(ret40.sub(med,axis=0))/(vol*np.sqrt(40)+1e-12)).shift(1)
ys={h:np.log(px.shift(-h)/px) for h in [1,5,10,20,40]}; qs={}
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
 q=qs[20][(qs[20].index>=a)&(qs[20].index<=b)]; print('regime',a.date(),b.date(),'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20320122_residual_reversal_40d_signal.csv',index=False)
