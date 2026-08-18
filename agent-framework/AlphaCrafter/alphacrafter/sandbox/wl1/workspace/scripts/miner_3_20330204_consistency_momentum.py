import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2033-02-03'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None:
  d.date=pd.to_datetime(d.date); raw[s]=d[d.date<=cut].set_index('date').sort_index().close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff()
# consistency-adjusted momentum: medium trend rewarded only when daily direction is persistent
m20=np.log(px/px.shift(20)); m60=np.log(px/px.shift(60))
cons=r.gt(0).rolling(40,min_periods=25).mean()-r.lt(0).rolling(40,min_periods=25).mean()
vol=r.rolling(40,min_periods=25).std()
f=(m20.rank(axis=1,pct=True)*.35+m60.rank(axis=1,pct=True)*.35+cons.rank(axis=1,pct=True)*.30).shift(1)
fr=np.log(px.shift(-10)/px); ics=[];ns=[];turn=[]; dates=[]
for i,d in enumerate(f.index):
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));dates.append(d)
 if i:
  z=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turn.append((z.iloc[:,0].rank()-z.iloc[:,1].rank()).abs().mean()/len(z))
s=pd.Series(ics,index=dates).dropna();print('assets',len(raw),'calendar_dates',len(px),'valid_dates',len(s),'avgN',np.mean(ns),'coverage',np.mean(np.array(ns)/15),'IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',np.mean(s>0),'turn',np.mean(turn))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2032','2033')]:
 q=s[(s.index.year>=int(a))&(s.index.year<=int(b))];print(a,b,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std() if len(q)>1 else np.nan)
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20330204_consistency_momentum_signal.csv',index=False)
