import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P='../persistent/stock_data/'
px={}
for s in U:
 d=pd.read_csv(P+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 px[s]=d['close']
C=pd.concat(px,axis=1).sort_index()
r=C.pct_change()
bread=(r<0).sum(axis=1)/r.notna().sum(axis=1)
# lagged breadth visible at decision; factor on t based on t-1 returns, forward return t+1..t+5
ret3=C.pct_change(3).shift(1)
# robust cross-sectional median centered negative return
f=-(ret3.sub(ret3.median(axis=1),axis=0))
fwd=C.shift(-5)/C-1
# rolling threshold uses only history through t-1
q=bread.shift(1).rolling(252,min_periods=60).quantile(.75)
thresholds={'adaptive75':np.maximum(.60,q),'adaptive_median':np.maximum(.60,bread.shift(1).rolling(252,min_periods=60).median()),'fixed70':pd.Series(.70,index=C.index)}
for name,th in thresholds.items():
  active=bread.shift(1)>=th
  vals=[]; dates=[]; ns=[]
  for dt in C.index:
   if not active.get(dt,False): continue
   x=f.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
   if len(z)>=8:
    vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); ns.append(len(z))
  a=np.array(vals); ic=a.mean() if len(a) else np.nan; icir=ic/a.std(ddof=1)*np.sqrt(1) if len(a)>1 else np.nan
  # 5 day ICIR convention mean/std daily ICs
  print(name,'dates',len(a),'active',int(active.sum()),'avgN',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(ic,ic/a.std(ddof=1) if len(a)>1 else np.nan,(a>0).mean() if len(a) else np.nan))
  for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2027')]:
   aa=np.array([v for v,d in zip(vals,dates) if lo<=str(d.year)<=hi])
   print(' ',lo,hi,'n',len(aa),'ic',aa.mean() if len(aa) else np.nan)
print('range',C.index.min(),C.index.max(),'dates',len(C),'instruments',C.shape[1])
