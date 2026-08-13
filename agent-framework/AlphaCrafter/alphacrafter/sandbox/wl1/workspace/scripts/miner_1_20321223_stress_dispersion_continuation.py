import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-12-22'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); raw[s]=d[d.date<=cut].set_index('date').sort_index().close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); res=r.sub(r.mean(axis=1),axis=0)
bench=r.mean(axis=1).rolling(20,min_periods=15).sum(); disp=res.std(axis=1); med=disp.rolling(120,min_periods=60).median(); vol=res.rolling(30,min_periods=20).std().replace(0,np.nan)
# Stress-conditioned continuation: lagged relative 10d momentum, activated by bearish benchmark or unusually high dispersion.
gate=((bench<0)|(disp>med)).shift(1).fillna(False).astype(float)
f=(res.rolling(10,min_periods=10).sum()/vol).shift(1).mul(gate,axis=0)
fr=np.log(px.shift(-10)/px); ics=[]; ns=[]; turn=[]
for i,d in enumerate(f.index):
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 if i:
  z=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
  if len(z)>=8: turn.append(z.iloc[:,0].rank().sub(z.iloc[:,1].rank()).abs().mean()/len(z))
s=pd.Series(ics).dropna(); print('assets',len(raw),'valid',len(s),'avgN',round(np.mean(ns),3),'coverage',round(np.mean(np.array(ns)/15),5),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round(np.mean(s>0),4),'turn',round(np.mean(turn),6),'active',round(float((f.abs().sum(axis=1)>0).mean()),4))
for a,b in [('2020','2024'),('2025','2028'),('2029','2030'),('2031','2032'),('2032','2032')]:
 q=[]
 for d in f.index:
  if a<=str(d)[:4]<=b:
   z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
   if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(q).dropna(); print(a,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(),6) if len(q)>1 else None)
out=f.where(f!=0).stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20321223_stress_dispersion_continuation_signal.csv',index=False)
