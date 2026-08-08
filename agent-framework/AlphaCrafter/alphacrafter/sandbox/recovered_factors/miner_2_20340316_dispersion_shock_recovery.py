import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().close for a in A}).sort_index().astype(float)
r=p.pct_change(); vol=r.rolling(20,min_periods=15).std()
# Novel idea: broad cross-asset dispersion shock followed by short relative recovery.
# Dispersion is cross-sectional daily return std; signal is lagged 3d relative reversal,
# activated when lagged dispersion is in its trailing 80d upper quartile.
disp=r.std(axis=1); q=disp.rolling(80,min_periods=40).quantile(.75)
active=(disp>q).astype(float).shift(1)
rel=r.rolling(3,min_periods=3).sum().sub(r.rolling(3,min_periods=3).sum().mean(axis=1),axis=0)
f=(-rel.div(vol.median(axis=1),axis=0)).mul(active,axis=0).shift(1)
print('ASOF',p.index.max().date(),'DATES',len(p),'ASSETS',len(A),'ACTIVE_DAYS',int(active.sum()))
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; ss=[]; ns=[]; ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   ss.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); ds.append(dt)
 s=pd.Series(ss,index=ds)
 print('H',h,'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'DATES',len(s),'MEAN_N',round(np.mean(ns),2),'HIT',round((s>0).mean(),4))
 for lo,hi in [('2020','2024'),('2024','2028'),('2028','2031'),('2031','2035')]:
  x=s[(s.index>=lo)&(s.index<hi)]
  if len(x): print(' REG',lo,round(x.mean(),6),round(x.mean()/x.std(ddof=1),6),len(x))
print('COVERAGE',round(f.notna().mean().mean(),6),'TURN10',round(f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean(),6))
