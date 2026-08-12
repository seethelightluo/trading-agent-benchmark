import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-05-26'); raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); d=d[d.date<=cut].set_index('date').sort_index(); raw[s]=d.close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); bench=r.mean(axis=1); resid=r.sub(bench,axis=0)
rv=resid.rolling(20,min_periods=15).std(); shock=resid.rolling(3,min_periods=3).sum()/rv
trend=resid.rolling(40,min_periods=30).sum(); base=(-shock).shift(1); disp=resid.std(axis=1).rolling(20,min_periods=15).mean(); high=disp>disp.rolling(120,min_periods=60).median(); weak=bench.rolling(20,min_periods=15).sum()<0
reg=(weak|high).shift(1).fillna(False); cond=trend.gt(0).shift(1).fillna(False).mul(reg,axis=0).astype(bool); f=base.where(cond); fr=np.log(px.shift(-10)/px)
vals=[]; ns=[]; turns=[]; dates=[]
for i,dt in enumerate(f.index):
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); dates.append(dt)
 if i:
  q=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
  if len(q)>=8: turns.append(q.iloc[:,0].rank().sub(q.iloc[:,1].rank()).abs().mean()/len(q))
s=pd.Series(vals,index=pd.DatetimeIndex(dates)); print('assets',len(raw),'dates',len(px),'valid',len(s),'avgN',np.mean(ns) if ns else 0,'active_frac',float(f.notna().any(axis=1).mean())); print('IC',s.mean() if len(s) else np.nan,'ICIR',s.mean()/s.std() if len(s)>1 else np.nan,'hit',np.mean(s>0) if len(s) else np.nan,'turn',np.mean(turns) if turns else np.nan)
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032')]:
 q=s[(s.index.year>=int(a))&(s.index.year<=int(b))]; print(a,b,len(q),q.mean() if len(q) else np.nan,q.mean()/q.std() if len(q)>1 else np.nan)
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20320527_trend_confirmed_shock_signal.csv',index=False)
