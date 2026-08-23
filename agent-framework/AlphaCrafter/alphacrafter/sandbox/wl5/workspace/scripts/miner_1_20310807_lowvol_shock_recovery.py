import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).sort_values('date').set_index('date')
 p[s]=d.close.replace(0,np.nan)
p=pd.DataFrame(p).sort_index(); r=p.pct_change()
# low-risk persistence: inverse 30d realized volatility, with a 5d shock-recovery tilt
vol=r.rolling(30,min_periods=15).std(); shock=r.rolling(5,min_periods=5).sum()
fac=(1/(vol+1e-8)) * (1 + 0.20*(-shock/(r.rolling(60,min_periods=20).std()+1e-8)).clip(-2,2))
fwd=p.shift(-5)/p-1
for h in [1,5,10,20]:
 ff=p.shift(-h)/p-1; out=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8: out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(out).dropna(); print('h',h,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
# coverage and turnover
valid=fac.notna().sum(axis=1); print('meanN',valid.mean(),'coverage',valid.mean()/15,'turn',fac.rank(pct=True,axis=1).diff().abs().mean(axis=1).mean())
for lo,hi in [('2025','2027'),('2028','2029'),('2030','2031')]:
 out=[]
 for dt in fac.loc[lo:hi].index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(out); print('regime',lo, len(q),q.mean(),q.mean()/q.std(ddof=1))
