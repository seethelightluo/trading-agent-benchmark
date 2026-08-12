import pandas as pd,numpy as np
from pathlib import Path
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2031-10-30'
def ld(s):
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index(); return d[d.index<=cut]
p=pd.concat({s:ld(s) for s in assets},axis=1); r=np.log(p).diff(); vol=r.rolling(30,min_periods=20).std()*np.sqrt(30)
sig=(-np.log(p/p.shift(7))/vol).shift(1); f=np.log(p.shift(-10)/p)
rows=[]; prev=None; turns=[]; ns=[]
for dt in sig.index:
 x,y=sig.loc[dt],f.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  rows.append((dt,x[ok].corr(y[ok]))); ns.append(ok.sum()); rr=x.rank()
  if prev is not None: turns.append((rr[ok]-prev[ok]).abs().mean()/15)
  prev=rr
z=pd.Series(dict(rows)); print('7D volatility-scaled reversal'); print('dates',len(z),'avg_names',np.mean(ns),'coverage',np.mean(ns)/15,'IC',z.mean(),'ICIR',z.mean()/z.std(),'hit',(z>0).mean(),'turnover',np.mean(turns))
for name,a,b in [('2020-22','2020','2022'),('2023-25','2023','2025'),('2026-28','2026','2028'),('2029-31','2029','2031')]:
 q=z.loc[a:b]; print(name,len(q),q.mean(),q.mean()/q.std())
q=z.tail(120); print('recent120',q.mean(),q.mean()/q.std())
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20311030_reversal7_signal.csv',index=False)
