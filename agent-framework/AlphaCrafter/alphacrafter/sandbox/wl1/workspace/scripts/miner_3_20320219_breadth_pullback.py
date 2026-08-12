import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-02-19'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); d=d[d.date<=cut].sort_values('date'); raw[s]=d.set_index('date').close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); vol=r.rolling(30,min_periods=20).std()
# Candidate: cross-asset breadth-conditioned pullback. Contrarian 5/10d shock is
# emphasized after broad weakness, but attenuated after broad strength; lagged.
shock=(-.6*np.log(px/px.shift(5))-.4*np.log(px/px.shift(10)))/(vol*np.sqrt(7)+1e-9)
breadth=(r.rolling(20,min_periods=12).mean()>0).mean(axis=1)
# smooth centered breadth multiplier: 0.5..1.5, no future data
mult=(1.5-breadth).clip(.35,1.65)
f=(shock.mul(mult,axis=0)).shift(1)
fr={h:np.log(px.shift(-h)/px) for h in [1,5,10,20]}
def calc(h):
 vals=[]; ns=[]; turns=[]
 for i,dt in enumerate(f.index):
  z=pd.concat([f.loc[dt],fr[h].loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
  if i:
   q=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
   if len(q)>=8: turns.append(q.iloc[:,0].rank().sub(q.iloc[:,1].rank()).abs().mean()/len(q))
 s=pd.Series(vals); return len(s),np.mean(ns),s.mean(),s.mean()/s.std(),np.mean(s>0),np.mean(turns)
for h in [1,5,10,20]: print('h',h,'dates avg_n IC ICIR hit turnover',calc(h))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032')]:
 vals=[]
 for dt in f.index:
  if not(str(dt)[:4]>=a and str(dt)[:4]<=b): continue
  z=pd.concat([f.loc[dt],fr[10].loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 s=pd.Series(vals); print('regime',a,b,'dates',len(s),'IC',s.mean(),'ICIR',s.mean()/s.std() if len(s)>1 else np.nan)
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20320219_breadth_pullback_signal.csv',index=False); print('artifact',len(out),'assets',len(px.columns))
