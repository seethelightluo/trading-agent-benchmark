import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-07-21'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); raw[s]=d[d.date<=cut].set_index('date').sort_index().close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); bench=r.mean(axis=1); resid=r.sub(bench,axis=0)
rv=r.rolling(20,min_periods=15).std(); disp=resid.std(axis=1); threshold=disp.rolling(120,min_periods=60).quantile(.60); active=(bench.rolling(20,min_periods=15).sum()>0)&(disp>threshold)
shock=resid.rolling(5,min_periods=4).sum()/rv.rolling(20,min_periods=15).mean(); f=(-shock).shift(1).where(active.shift(1),0.0); fr=np.log(px.shift(-10)/px)
rows=[]; turns=[]
for i,dt in enumerate(f.index):
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 if i:
  q=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8: turns.append(q.iloc[:,0].rank().sub(q.iloc[:,1].rank()).abs().mean()/len(q))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); s=z.ic; print('assets',len(raw),'dates',len(px),'valid',len(s),'avgN',z.n.mean(),'coverage',z.n.mean()/15,'active_frac',float((f!=0).any(axis=1).mean())); print('IC',s.mean(),'ICIR',s.mean()/s.std(ddof=1),'hit',np.mean(s>0),'turn',np.mean(turns))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032')]:
 q=z.loc[a:b].ic; print(a,b,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20320722_trend_conditioned_pullback_signal.csv',index=False)
