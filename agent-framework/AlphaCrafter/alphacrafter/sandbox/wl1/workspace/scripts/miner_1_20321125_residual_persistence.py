import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-11-24'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); raw[s]=d[d.date<=cut].set_index('date').sort_index().close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); res=r.sub(r.mean(axis=1),axis=0)
# Risk-adjusted residual trend persistence: recent residual trend rewarded only when persistent.
trend=res.rolling(40,min_periods=30).sum(); persistence=(res>0).rolling(40,min_periods=30).mean(); vol=res.rolling(40,min_periods=30).std(); f=(trend*persistence/(vol+1e-8)).shift(1)
fr=np.log(px.shift(-10)/px); ics=[];ns=[];turn=[]
for i,d in enumerate(f.index):
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 if i:
  z=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
  if len(z)>=8: turn.append(z.iloc[:,0].rank().sub(z.iloc[:,1].rank()).abs().mean()/len(z))
s=pd.Series(ics).dropna(); print('factor risk_adjusted_residual_persistence_40d'); print('assets',len(raw),'dates',len(px),'valid',len(s),'avgN',np.mean(ns),'coverage',np.mean(np.array(ns)/len(U)),'IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',np.mean(s>0),'turn',np.mean(turn))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032')]:
 q=[]
 for d in f.index:
  if a<=str(d)[:4]<=b:
   z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
   if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(q).dropna(); print(a,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std() if len(q)>1 else np.nan)
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20321125_residual_persistence_signal.csv',index=False)
