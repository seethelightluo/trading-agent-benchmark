import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-10-27'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); raw[s]=d[d.date<=cut].set_index('date').sort_index().close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); res=r.sub(r.mean(axis=1),axis=0)
# Shock reversal: buy assets with unusually poor recent residual performance during stressed dispersion.
vol=res.rolling(20,min_periods=15).std(); disp=res.std(axis=1).rolling(20,min_periods=15).mean()
stress=(disp>disp.rolling(120,min_periods=60).quantile(.65)).shift(1).fillna(False)
z=(res.rolling(5,min_periods=5).sum()/vol.rolling(20,min_periods=15).mean()).shift(1)
f=-z.mul(stress.astype(float),axis=0)
fr=np.log(px.shift(-10)/px); ics=[]; ns=[]; turns=[]
for i,d in enumerate(f.index):
 q=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(q)>=8: ics.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman')); ns.append(len(q))
 if i:
  q=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
  if len(q)>=8: turns.append(q.iloc[:,0].rank().sub(q.iloc[:,1].rank()).abs().mean()/len(q))
s=pd.Series(ics).dropna(); print('assets',len(raw),'dates',len(px),'valid',len(s),'avgN',np.mean(ns),'coverage',np.mean([n/len(U) for n in ns])); print('IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',np.mean(s>0),'turn',np.mean(turns),'active',float((f.abs().sum(axis=1)>0).mean()))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2031','2032')]:
 q=[]
 for d in f.index:
  if a<=str(d)[:4]<=b:
   x=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
   if len(x)>=8:q.append(x.iloc[:,0].corr(x.iloc[:,1],method='spearman'))
 q=pd.Series(q).dropna(); print(a,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std() if len(q)>1 else np.nan)
out=f.where(f!=0).stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20321028_shock_reversal_signal.csv',index=False)
