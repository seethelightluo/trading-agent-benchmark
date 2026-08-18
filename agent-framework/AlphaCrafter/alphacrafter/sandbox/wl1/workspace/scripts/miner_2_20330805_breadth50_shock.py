import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2033-08-04'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None:
  d.date=pd.to_datetime(d.date); raw[s]=d[d.date<=cut].set_index('date').sort_index().close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); resid=r.sub(r.mean(axis=1),axis=0); shock=resid.rolling(5,min_periods=4).sum(); rv=resid.rolling(30,min_periods=15).std(); breadth=(r>0).mean(axis=1)
f=(-shock/(rv*np.sqrt(5)+1e-8)).where(breadth.rolling(10,min_periods=7).mean()<.50,0).shift(1); fr=np.log(px.shift(-10)/px)
ics=[]; ns=[]; ds=[]; ts=[]
for i,d in enumerate(f.index):
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); ds.append(d)
 if i:
  q=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
  if len(q)>=8: ts.append(q.iloc[:,0].rank().sub(q.iloc[:,1].rank()).abs().mean()/len(q))
s=pd.Series(ics,index=pd.to_datetime(ds)).dropna(); print('assets',len(raw),'dates',len(px),'valid',len(s),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(np.array(ns)/15),4),'active',round((breadth.rolling(10,min_periods=7).mean()<.5).mean(),4),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round((s>0).mean(),4),'turn',round(np.mean(ts),4))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2032','2033')]:
 q=s[(s.index.year>=int(a))&(s.index.year<=int(b))]; print(a,b,len(q),round(q.mean(),6),round(q.mean()/q.std(),6) if len(q)>1 else np.nan)
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20330805_breadth50_shock_signal.csv',index=False)
