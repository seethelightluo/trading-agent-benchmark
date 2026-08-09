import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date') for a in assets}
cl=pd.DataFrame({a:D[a]['close'] for a in assets}).sort_index(); r=np.log(cl).diff()
# one interpretable idea: cross-sectional relative volatility shock, lagged
v=r.rolling(20,min_periods=15).std(); x=v.div(v.median(axis=1),axis=0); sig=-np.log(x).shift(1)
print('data',cl.index.min(),cl.index.max(),'assets',len(assets),'coverage',sig.notna().mean().mean())
for h in [1,5,10,20]:
 fwd=cl.shift(-h)/cl-1; ics=[]; dates=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): ics.append(q); dates.append(dt)
 a=np.array(ics); print('H',h,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'dates',len(a),'meanN',np.mean([len(pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()) for d in dates]))
# turnover 10d rank
s=sig.rank(axis=1,pct=True); print('turn',np.nanmean(abs(s-s.shift(10)).sum(axis=1)/s.notna().sum(axis=1)))
# regime h10
fwd=cl.shift(-10)/cl-1
for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2034')]:
 aa=[]
 for dt in sig.loc[lo:hi].index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(aa); print('reg',lo,hi,a.mean(),a.mean()/a.std(ddof=1),len(a))
