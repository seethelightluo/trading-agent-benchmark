import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-11-19')
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index() for a in U}
D={a:d[d.index<=end] for a,d in D.items()}
# smoothed 3-event overnight reversal
fac=pd.concat({a:-(d.open/d.close.shift(1)-1).rolling(3,min_periods=2).mean() for a,d in D.items()},axis=1)
cl=pd.concat({a:d.close for a,d in D.items()},axis=1)
for h in [1,5,10]:
 fwd=cl.pct_change(h).shift(-h); ics=[]; ns=[]; dates=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));dates.append(dt)
 s=pd.Series(ics,index=dates);print('H',h,'dates',len(s),'avgN',np.mean(ns),'IC',s.mean(),'ICIR',s.mean()/s.std(ddof=1),'hit',(s>0).mean())
 if h==1:
  for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
   z=s[(s.index.year>=lo)&(s.index.year<=hi)]; print('REG',lo,hi,len(z),z.mean())
print('coverage',fac.notna().mean().mean(),'dates',fac.index.nunique())
r=fac.rank(axis=1,pct=True);print('turnover',r.diff().abs().mean().mean())
fac.stack().rename('signal').reset_index().rename(columns={'level_1':'symbol'}).to_csv('scripts/miner_2_20261119_overnight_gap_signal.csv',index=False)
