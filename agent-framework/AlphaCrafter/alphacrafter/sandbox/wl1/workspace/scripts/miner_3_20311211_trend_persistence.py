import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2031-12-10'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); d=d[d.date<=cut].sort_values('date'); raw[s]=d.set_index('date').close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff()
# Trend persistence: medium momentum, rewarded when daily direction is consistent; risk-adjust by downside deviation.
mom=np.log(px/px.shift(40)); persist=(r.rolling(40).mean().abs()+1e-9)**0 # placeholder
pos=r.gt(0).rolling(40).mean(); down=r.where(r<0,0).pow(2).rolling(40,min_periods=20).mean().pow(.5)
f=(mom*(0.5+pos))/(down+1e-9); f=f.shift(1)
ics=[]; obs=[]; turns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],np.log(px.shift(-20)/px).loc[dt]],axis=1).dropna()
 if len(z)>=8: ics.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))); obs.append(len(z))
 if dt in f.index[1:]:
  prev=f.index[f.index.get_loc(dt)-1]; q=pd.concat([f.loc[prev],f.loc[dt]],axis=1).dropna()
  if len(q)>=8: turns.append(q.iloc[:,0].rank().sub(q.iloc[:,1].rank()).abs().mean()/len(q))
ics=pd.Series(dict(ics)).dropna(); print('dates',len(ics),'avg_n',np.mean(obs),'coverage',np.mean(obs)/15,'IC20',ics.mean(),'ICIR',ics.mean()/ics.std(),'hit',np.mean(ics>0),'turn',np.nanmean(turns))
for h in [1,5,10,20]:
 v=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],np.log(px.shift(-h)/px).loc[dt]],axis=1).dropna()
  if len(z)>=8:v.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(v))
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2031')]:
 q=ics[(ics.index>=a)&(ics.index<=b)];print('regime',a,b,len(q),q.mean(),q.mean()/q.std() if len(q)>1 else np.nan)
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20311211_trend_persistence_signal.csv',index=False);print('artifact',len(out))
