import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-04-28'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); d=d[d.date<=cut].set_index('date').sort_index(); raw[s]=d.close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); bench=r.mean(axis=1); resid=r.sub(bench,axis=0)
vol=r.rolling(20,min_periods=15).std(); base=(-resid.rolling(10,min_periods=8).sum()/vol.rolling(20,min_periods=15).mean()).shift(1)
disp=resid.std(axis=1).rolling(20,min_periods=15).mean(); high=(disp>disp.rolling(120,min_periods=60).median()); stress=bench.rolling(20,min_periods=15).sum()<0
f=base.where((stress|high).shift(1));
for h in [1,5,10,20]:
 fr=np.log(px.shift(-h)/px); vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 s=pd.Series(vals); print('h',h,'dates',len(s),'avgN',np.mean(ns),'IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',np.mean(s>0))
fr=np.log(px.shift(-10)/px); turns=[]
for i in range(1,len(f)):
 q=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
 if len(q)>=8: turns.append(q.iloc[:,0].rank().sub(q.iloc[:,1].rank()).abs().mean()/len(q))
print('turn',np.mean(turns),'active',f.notna().any(axis=1).mean())
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20320429_stress_or_dispersion_signal.csv',index=False)
