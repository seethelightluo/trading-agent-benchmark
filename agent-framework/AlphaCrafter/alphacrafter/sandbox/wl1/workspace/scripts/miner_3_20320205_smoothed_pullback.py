import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-02-04'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  d=d.copy();d.date=pd.to_datetime(d.date);d=d[d.date<=cut].sort_values('date');raw[s]=d.set_index('date').close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); vol=r.rolling(30,min_periods=20).std()
# Lower-turnover smoothed pullback: average 5d and 10d normalized shocks, gated by a persistent 80d positive trend.
shock=(-0.60*np.log(px/px.shift(5))-0.40*np.log(px/px.shift(10)))/(vol*np.sqrt(7)+1e-9)
gate=(np.tanh(np.log(px/px.shift(80))*2)+1)/2
f=(shock*gate).shift(1)
def run(h):
 vals=[]; ns=[]; turns=[]
 fr=np.log(px.shift(-h)/px)
 for i,dt in enumerate(f.index):
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
  if i:
   q=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
   if len(q)>=8: turns.append(q.iloc[:,0].rank().sub(q.iloc[:,1].rank()).abs().mean()/len(q))
 s=pd.Series(dict(vals)); return s,np.mean(ns),np.mean(turns)
for h in [1,5,10,20]:
 s,n,t=run(h);print('h',h,'dates',len(s),'avg_n',n,'coverage',n/15,'IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',np.mean(s>0),'turn',t)
s,n,t=run(10)
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032')]:
 q=s[(s.index>=a)&(s.index<=b)];print('regime',a,b,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std() if len(q)>1 else np.nan)
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20320205_smoothed_pullback_signal.csv',index=False);print('artifact',len(out))
