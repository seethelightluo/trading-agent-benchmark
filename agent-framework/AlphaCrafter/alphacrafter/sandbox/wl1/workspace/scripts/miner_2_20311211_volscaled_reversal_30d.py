import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2031-12-10'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000); d['date']=pd.to_datetime(d.date); d=d[d.date<=cut].sort_values('date'); raw[s]=d.set_index('date').close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff();
# Volatility-scaled intermediate-horizon reversal, lagged one day.
f=(-np.log(px/px.shift(30))/(r.rolling(60).std()+1e-9)).shift(1)
fw={h:np.log(px.shift(-h)/px) for h in [1,5,10,20]}
allq={}
for h,x in fw.items():
 vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],x.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 q=pd.Series(dict(vals)).dropna();allq[h]=q
 print('horizon',h,'dates',len(q),'avg_n',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6),'hit',round(np.mean(q>0),4))
for a,b in [('2020-01-01','2025-12-31'),('2026-01-01','2028-12-31'),('2029-01-01','2030-12-31'),('2031-01-01','2031-12-10')]:
 q=allq[20].loc[a:b];print('regime',a,b,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6))
# rank turnover and artifact
rank=f.rank(axis=1,pct=True); turn=(rank.diff().abs().mean(axis=1)).dropna();print('turnover',round(turn.mean(),6))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20311211_volscaled_reversal_30d_signal.csv',index=False)
