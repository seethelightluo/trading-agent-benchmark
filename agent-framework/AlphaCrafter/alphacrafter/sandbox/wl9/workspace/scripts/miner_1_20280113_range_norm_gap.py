import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; R=Path('../persistent/stock_data'); ds={}
for a in A:
 d=pd.read_csv(R/(a+'.csv'),parse_dates=['date']).sort_values('date'); r=d.close.pct_change(); d['f']=-((d.open/d.close.shift(1)-1)/(r.rolling(20).std()+1e-12)); ds[a]=d.set_index('date')
for h in [1,5,10]:
 ics=[]; ns=[]
  for dt in sorted(set().union(*[set(d.index) for d in ds.values()])):
   v=[]; y=[]
   for a,d in ds.items():
    if dt not in d.index: continue
    i=d.index.get_loc(dt)
    if i+h>=len(d) or i<20: continue
    if np.isfinite(d.iloc[i].f): v.append(d.iloc[i].f); y.append(d.iloc[i+h].close/d.iloc[i].close-1)
   if len(v)>=8:
    q=spearmanr(v,y).statistic
    if np.isfinite(q): ics.append(q);ns.append(len(v))
  z=pd.Series(ics);print('h',h,'dates',len(z),'avgN',np.mean(ns),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean(),'coverage',np.mean(ns)/15)
