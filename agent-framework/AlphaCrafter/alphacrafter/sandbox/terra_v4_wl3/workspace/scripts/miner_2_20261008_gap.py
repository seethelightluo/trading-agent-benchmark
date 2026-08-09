import pandas as pd,numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-10-08')
D={};
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index();D[a]=d[d.index<=end]
# gap shock: overnight open versus prior close, smoothed 3 events; contrarian gap tends to mean revert
fac=pd.concat({a:-(D[a].open/D[a].close.shift(1)-1).rolling(3,min_periods=2).mean() for a in assets},axis=1)
cl=pd.concat({a:D[a].close for a in assets},axis=1)
for h in [1,5,10]:
 fwd=cl.pct_change(h).shift(-h);ics=[];ds=[];ns=[]
 for dt in fac.index:
  x=fac.loc[dt].dropna();y=fwd.loc[dt].reindex(x.index).dropna();x=x.reindex(y.index)
  if len(x)>=8 and x.nunique()>1 and y.nunique()>1:ics.append(spearmanr(x,y).statistic);ds.append(dt);ns.append(len(x))
 s=pd.Series(ics,index=ds);print('H',h,'dates',len(s),'avgN',np.mean(ns),'IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',(s>0).mean())
 if h==1:
  for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
   z=s[(s.index.year>=lo)&(s.index.year<=hi)];print('REG',lo,hi,len(z),z.mean(),z.mean()/z.std())
print('coverage',fac.notna().sum(axis=1).mean()/15,'turnover',fac.rank(pct=True).diff().abs().mean(axis=1).mean())
out=fac.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20261008_gap_signal.csv',index=False)
