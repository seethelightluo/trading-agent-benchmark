import pandas as pd,numpy as np
from pathlib import Path
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2031-11-12'
def get(s,col):
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index(); return d[col].reindex(pd.date_range(d.index.min(),pd.Timestamp(cut),freq='D'))
p=pd.concat({s:get(s,'close') for s in assets},axis=1); v=pd.concat({s:get(s,'volume') for s in assets},axis=1)
r=np.log(p).diff(); rv=r.rolling(20,min_periods=12).std(); volshock=np.log(v.replace(0,np.nan)).diff(5).clip(-3,3)
# Reversal is activated by unusually high volume, with risk normalization.
sig=(-np.log(p/p.shift(3))/rv* (1+0.35*volshock.clip(lower=0))).shift(1); fwd=np.log(p.shift(-10)/p)
rows=[]; ns=[]; prev=None; turns=[]
for dt in sig.index:
 x,y=sig.loc[dt],fwd.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  rows.append((dt,x[ok].corr(y[ok]))); ns.append(ok.sum()); rank=x.rank()
  if prev is not None: turns.append((rank[ok]-prev[ok]).abs().mean()/15)
  prev=rank
z=pd.Series(dict(rows)); print('dates',len(z),'avg_names',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(z.mean(),5),'ICIR',round(z.mean()/z.std(),5),'hit',round((z>0).mean(),4),'turnover',round(np.mean(turns),5))
for lo,hi in [('2026-01-01','2028-12-31'),('2029-01-01','2031-11-12')]:
 q=z.loc[lo:hi]; print(lo[:4]+'-'+hi[:4],len(q),round(q.mean(),5),round(q.mean()/q.std(),5))
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20311113_volume_activated_reversal_signal.csv',index=False)
