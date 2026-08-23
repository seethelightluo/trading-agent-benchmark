import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv(Path('../persistent/stock_data')/(a+'.csv'),parse_dates=['date']).set_index('date')['close'] for a in A}
p=pd.DataFrame(D).ffill().loc[:'2027-04-07']; r=p.pct_change()
# Bear-breadth conditioned 5d reversal; all inputs lagged by one completed day
r5=p.pct_change(5).shift(1); breadth=r5.median(axis=1)
active=breadth < -0.01
f=(-r5).where(active, np.nan)
# use cross-sectional ranks on active dates
stats=[]
for h in [1,5,10]:
 fr=p.shift(-h).div(p)-1; vals=[]; ns=[]; ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); ds.append(dt)
 s=pd.Series(vals,index=ds).dropna(); stats.append((h,len(s),np.mean(ns),s.mean(),s.std(ddof=1),s.mean()/s.std(ddof=1)*np.sqrt(len(s)),(s>0).mean()))
print('period',p.index.min().date(),p.index.max().date(),'assets',len(A),'active_days',active.sum())
for x in stats: print('horizon valid_dates avg_n IC ICIR hit',x)
print('conditional coverage',f.notna().sum(axis=1).ge(8).mean(),'rank_turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
for nm,(a,b) in {'2020-22':('2020','2022-12-31'),'2023-24':('2023','2024-12-31'),'2025-26':('2025','2026-12-31'),'2027':('2027','2027-04-07')}.items():
 s=[]
 for dt in f.loc[a:b].index:
  z=pd.concat([f.loc[dt],(p.shift(-1).div(p)-1).loc[dt]],axis=1).dropna()
  if len(z)>=8:s.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(nm,len(s),np.mean(s) if s else np.nan,(np.mean(s)/np.std(s,ddof=1)*np.sqrt(len(s)) if len(s)>1 else np.nan))
