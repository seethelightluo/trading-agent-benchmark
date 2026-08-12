import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2031-12-24'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); d=d[d.date<=cut].sort_values('date'); raw[s]=d.set_index('date').close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff()
# Breadth-conditioned short-term reversal: fade 10d relative moves, with a smooth penalty
# when the broad cross-asset tape is strongly trending (breadth confirmation).
m10=np.log(px/px.shift(10)); breadth=r.rolling(20).mean().gt(0).mean(axis=1)
# factor rewards lagged losers, but less aggressively during one-sided breadth regimes
reg=(1-2*(breadth-.5).abs()).clip(.25,1.0)
f=(-m10).mul(reg,axis=0).shift(1)
fw=np.log(px.shift(-20)/px); vals=[]; ns=[]; turns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))); ns.append(len(z))
for i in range(1,len(f)):
 q=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
 if len(q)>=8: turns.append((q.iloc[:,0].rank()-q.iloc[:,1].rank()).abs().mean()/len(q))
ics=pd.Series(dict(vals)).dropna(); print('dates',len(ics),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15,'IC20',ics.mean(),'ICIR',ics.mean()/ics.std(),'hit',np.mean(ics>0),'turn',np.nanmean(turns))
for h in [1,5,10,20]:
 v=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],np.log(px.shift(-h)/px).loc[dt]],axis=1).dropna()
  if len(z)>=8:v.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(v))
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2031')]:
 q=ics[(ics.index>=a)&(ics.index<=b)]; print('regime',a,b,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std() if len(q)>1 else np.nan)
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20311225_breadth_conditioned_reversal_signal.csv',index=False); print('artifact',len(out))
