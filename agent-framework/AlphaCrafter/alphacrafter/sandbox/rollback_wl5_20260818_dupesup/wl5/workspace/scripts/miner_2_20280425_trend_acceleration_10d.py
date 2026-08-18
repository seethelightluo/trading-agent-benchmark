import os, json
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

UNIVERSE=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Trend acceleration: recent 10d return relative to prior 20d return, with volatility normalization.
frames={}
for s in UNIVERSE:
    d=get_stock_daily_data(s, days=2200)
    if d is not None and len(d)>100:
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.sort_values('date').set_index('date')
        frames[s]=d['close'].astype(float)
px=pd.concat(frames,axis=1).sort_index().ffill()
ret=px.pct_change()
# acceleration is recent 10d return minus preceding 20d return; divide by trailing 20d vol
sig=(px/px.shift(10)-1) - (px.shift(10)/px.shift(30)-1)
vol=ret.rolling(20).std()*np.sqrt(252)
sig=sig/vol.replace(0,np.nan)
rows=[]
for dt in sig.index:
    x=sig.loc[dt]
    for h in [5,10,20]:
        y=px.shift(-h).loc[dt]/px.loc[dt]-1
        z=pd.concat([x,y],axis=1).dropna()
        if len(z)>=8:
            ic=z.iloc[:,0].rank().corr(z.iloc[:,1].rank())
            rows.append((dt,h,ic,len(z)))
r=pd.DataFrame(rows,columns=['date','h','ic','n'])
print('data_dates',px.index.min(),px.index.max(),'assets',len(frames),'rows',len(r))
for h in [5,10,20]:
 q=r[r.h==h]; print('H',h,'obs',len(q),'meanN',q.n.mean(),'IC %.6f ICIR %.6f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1), (q.ic>0).mean()))
 for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2099')]:
  q2=q[(q.date>=a)&(q.date<=b)]
  if len(q2): print(' ',a,b,len(q2),'IC %.6f ICIR %.6f'%(q2.ic.mean(),q2.ic.mean()/q2.ic.std(ddof=1)))
# rank turnover on daily signal, mean pairwise rank change
rr=sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna(); print('turnover',rr.mean(),'coverage',sig.notna().mean().mean())
# artifact for deterministic audit
out='scripts/miner_2_20280425_trend_acceleration_10d_signal.csv'
sig.to_csv(out,index_label='date')
print('artifact',out)
