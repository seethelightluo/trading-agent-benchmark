import pandas as pd,numpy as np
from pathlib import Path
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2031-11-12'
def ld(s):
 p=Path('../persistent/stock_data')/(s+'.csv'); d=pd.read_csv(p,parse_dates=['date']).set_index('date')['close'].sort_index(); return d[d.index<=cut]
p=pd.concat({s:ld(s) for s in assets},axis=1); r=np.log(p).diff(); vol=r.rolling(30,min_periods=20).std()*np.sqrt(30)
# Candidate: short reversal stabilized by a small medium-term trend confirmation.
base=-np.log(p/p.shift(7))/vol
trend=np.log(p/p.shift(40))/(r.rolling(60,min_periods=35).std()*np.sqrt(40))
fwd=np.log(p.shift(-10)/p)
for a in [0.10,0.20,0.30,0.40,0.50]:
 sig=(base+a*trend).shift(1); rows=[]; ns=[]; prev=None; turns=[]
 for dt in sig.index:
  x,y=sig.loc[dt],fwd.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8:
   rows.append((dt,x[ok].corr(y[ok]))); ns.append(ok.sum()); rank=x.rank()
   if prev is not None: turns.append((rank[ok]-prev[ok]).abs().mean()/15)
   prev=rank
 z=pd.Series(dict(rows)); print('alpha',a,'dates',len(z),'avg_names',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(z.mean(),5),'ICIR',round(z.mean()/z.std(),5),'hit',round((z>0).mean(),4),'turnover',round(np.mean(turns),5))
 for name,lo,hi in [('2020-22','2020','2022'),('2023-25','2023','2025'),('2026-28','2026','2028'),('2029-31','2029','2031')]:
  q=z.loc[lo:hi]; print(' regime',name,len(q),round(q.mean(),5),round(q.mean()/q.std(),5))
 if a==0.3:
  out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20311113_trend_confirmed_reversal_signal.csv',index=False)
