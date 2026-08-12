import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  d=d[['date','close']].copy(); d.date=pd.to_datetime(d.date); frames[s]=d.drop_duplicates('date').set_index('date').close
px=pd.concat(frames,axis=1).sort_index().ffill(); r=px.pct_change(); bench=r.mean(axis=1); resid=r.sub(bench,axis=0)
disp=r.std(axis=1); med=disp.rolling(120,min_periods=60).median(); act=((disp/med)-.85).clip(0,1.5)
results=[]
for w in [3,7,10]:
 f=(-resid.rolling(w,min_periods=w).sum()).div(resid.rolling(20,min_periods=15).std()).mul(act,axis=0).replace([np.inf,-np.inf],np.nan)
 y=px.shift(-10).div(px)-1; z=[]; ds=[]; ns=[]; prev=None; turns=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8:
   q=f.loc[dt,ok].corr(y.loc[dt,ok],method='spearman'); z.append(q); ds.append(dt); ns.append(ok.sum())
   rr=f.loc[dt,ok].rank(pct=True)
   if prev is not None: turns.append((rr-prev.reindex(rr.index)).abs().mean())
   prev=rr
 z=pd.Series(z,index=pd.to_datetime(ds)).dropna(); ic=z.mean(); ir=ic/z.std(ddof=1)
 print('WINDOW',w,'dates',len(z),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15,'IC',ic,'ICIR',ir,'hit',np.mean(z>0),'turnover',np.mean(turns))
 for lo,hi in [('2024','2026'),('2027','2029'),('2030','2032')]:
  q=z.loc[lo:hi]; print('REGIME',lo,len(q),q.mean(),q.mean()/q.std(ddof=1))
 out=f.copy(); out.index.name='date'; out.to_csv(f'scripts/miner_2_20320902_residual_reversal_{w}d_signal.csv')
print('last',px.index.max())
