import pandas as pd,numpy as np
from pathlib import Path
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def l(s):
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index();return d[d.index<='2031-10-16']
p=pd.concat({s:l(s) for s in A},axis=1);r=np.log(p).diff();v=r.rolling(30,min_periods=20).std()*np.sqrt(30)
sig=(-np.log(p/p.shift(5))/v).shift(1); f=np.log(p.shift(-10)/p); z=[];n=[];t=[];pr=None
for d in sig.index:
 x,y=sig.loc[d],f.loc[d];o=x.notna()&y.notna()
 if o.sum()>=8:
  z.append((d,x[o].corr(y[o])));n.append(o.sum());q=x.rank()
  if pr is not None:t.append((q[o]-pr[o]).abs().mean()/15)
  pr=q
z=pd.Series(dict(z));print('5D volscaled reversal',len(z),np.mean(n),np.mean(n)/15,z.mean(),z.mean()/z.std(),(z>0).mean(),np.mean(t));print('recent',z.tail(120).mean(),z.tail(120).mean()/z.tail(120).std());print('2026+',z.loc['2026':].mean(),z.loc['2026':].mean()/z.loc['2026':].std())
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20311016_reversal5_signal.csv',index=False)
