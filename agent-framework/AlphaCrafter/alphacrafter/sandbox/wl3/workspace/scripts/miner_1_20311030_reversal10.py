import pandas as pd,numpy as np
from pathlib import Path
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2031-10-30'
def ld(s): return pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index().loc[:cut]
p=pd.concat({s:ld(s) for s in A},axis=1); r=np.log(p).diff(); v=r.rolling(40,min_periods=25).std()*np.sqrt(40)
s=(-np.log(p/p.shift(10))/v).shift(1); y=np.log(p.shift(-10)/p); rows=[]; ns=[]; prev=None; tr=[]
for d in s.index:
 x=s.loc[d]; q=y.loc[d]; ok=x.notna()&q.notna()
 if ok.sum()>=8:
  rows.append((d,x[ok].corr(q[ok]))); ns.append(ok.sum()); rank=x.rank()
  if prev is not None: tr.append((rank[ok]-prev[ok]).abs().mean()/15)
  prev=rank
z=pd.Series(dict(rows)); print('10D reversal'); print('dates',len(z),'avg_names',np.mean(ns),'coverage',np.mean(ns)/15,'IC',z.mean(),'ICIR',z.mean()/z.std(),'hit',(z>0).mean(),'turnover',np.mean(tr))
for n,a,b in [('2020-22','2020','2022'),('2023-25','2023','2025'),('2026-28','2026','2028'),('2029-31','2029','2031')]:
 q=z.loc[a:b]; print(n,len(q),q.mean(),q.mean()/q.std())
q=z.tail(120); print('recent120',q.mean(),q.mean()/q.std())
out=s.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20311030_reversal10_signal.csv',index=False)
