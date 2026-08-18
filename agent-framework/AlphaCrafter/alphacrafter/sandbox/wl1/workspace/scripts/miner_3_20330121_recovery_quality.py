import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2033-01-20'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); raw[s]=d[d.date<=cut].set_index('date').sort_index().close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff()
# Candidate: short-horizon rebound quality. Reward recent 5d recovery, but penalize
# 20d downside volatility and require a positive/less-negative 40d trend context.
down20=r.where(r<0,0).rolling(20,min_periods=12).std()
short=np.log(px/px.shift(5))/down20.replace(0,np.nan)
context=np.log(px/px.shift(40))
# cross-sectional ranks, with context gating: recovery is favored only when context is not bottom quartile
base=short.rank(axis=1,pct=True)
gate=(context.rank(axis=1,pct=True).clip(lower=.25)-.25)/.75
f=(base*(0.65+0.35*gate)).shift(1)
fr=np.log(px.shift(-10)/px); ics=[]; ns=[]; turn=[]
for i,d in enumerate(f.index):
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 if i:
  z=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
  if len(z)>=8: turn.append(z.iloc[:,0].rank().sub(z.iloc[:,1].rank()).abs().mean()/len(z))
s=pd.Series(ics).dropna(); print('assets',len(raw),'calendar_dates',len(px),'valid_dates',len(s),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(np.array(ns)/15),4),'IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',np.mean(s>0),'turn',np.mean(turn))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2032','2033')]:
 q=[]
 for d in f.index:
  if a<=str(d)[:4]<=b:
   z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
   if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(q).dropna(); print(a,b,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std() if len(q)>1 else np.nan)
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20330121_recovery_quality_signal.csv',index=False)
