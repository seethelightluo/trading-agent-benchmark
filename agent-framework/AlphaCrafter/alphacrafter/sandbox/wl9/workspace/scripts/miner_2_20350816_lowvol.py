import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
a={}
for s in U:
 d=get_stock_daily_data(s,days=10000)
 if d is None or len(d)<150:d=get_index_daily_data(s,days=10000)
 if d is not None and len(d):
  x=d[['date','close']].copy();x.date=pd.to_datetime(x.date);a[s]=x.drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(a).sort_index();r=p.pct_change()
# low-volatility signal, lagged, with a mild positive carry from stable 20-day trend
v=r.rolling(60,min_periods=40).std()
sig=(-v + 0.15*p.pct_change(20)).shift(1).clip(-.5,.5)
for h in [5,10,20,40,60]:
 f=p.shift(-h)/p-1; z=[]; ds=[]; ns=[]
 for dt in sig.index:
  q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:z.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'));ds.append(dt);ns.append(len(q))
 z=pd.Series(z,index=pd.to_datetime(ds)).dropna();print(f'H{h} IC {z.mean():+.6f} ICIR {z.mean()/z.std(ddof=1):+.6f} hit {(z>0).mean():.4f} dates {len(z)} avgN {np.mean(ns):.2f}')
 if h==10:
  print('H10 coverage',sig.notna().mean().mean(),'turnover10',sig.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean())
  for lo,hi in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
   q=z[(z.index>=lo+'-01-01')&(z.index<=hi+'-12-31')]
   if len(q):print(f'REG {lo}-{hi} n {len(q)} IC {q.mean():+.6f} ICIR {q.mean()/q.std(ddof=1):+.6f}')
print('DATA',len(p),len(p.columns),p.index.min(),p.index.max())
sig.reset_index().to_csv('scripts/miner_2_20350816_lowvol_signal.csv',index=False)
