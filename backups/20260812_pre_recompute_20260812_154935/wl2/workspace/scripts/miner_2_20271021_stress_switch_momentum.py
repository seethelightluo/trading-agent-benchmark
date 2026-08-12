import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 p='../persistent/stock_data/'+s+'.csv'; x=pd.read_csv(p,parse_dates=['date']); x.date=x.date.dt.normalize(); return x.set_index('date').sort_index()
D={s:load(s) for s in U}; end=pd.Timestamp('2027-10-20')
dates=D['SPX'].index[(D['SPX'].index>='2020-04-01')&(D['SPX'].index<=end)]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); R=C.pct_change()
# Observation-only VIX, aligned and lagged to ensure only completed data.
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']); v.date=v.date.dt.normalize(); v=v.set_index('date').close.reindex(dates)
# Macro-switch: high stress uses short reversal; calm uses medium momentum.
stress=(v.shift(1)>v.shift(1).rolling(60,min_periods=40).median()).astype(float)
ret5=R.rolling(5,min_periods=5).sum(); ret20=R.rolling(20,min_periods=20).sum(); vol=R.rolling(20,min_periods=20).std()
# cross-sectional demean and volatility normalization, lagged one day
rev=-ret5.div(vol).sub((-ret5.div(vol)).median(axis=1),axis=0)
mom=ret20.div(vol).sub((ret20.div(vol)).median(axis=1),axis=0)
F=rev.mul(stress,axis=0)+mom.mul(1-stress,axis=0)
F=F.shift(1)
for h in [1,3,5,10]:
 Y=C.shift(-h).div(C)-1; a=[]; ns=[]; ds=[]
 for dt in dates:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   a.append(spearmanr(z.f,z.y).statistic); ns.append(len(z)); ds.append(dt)
 a=np.asarray(a); print('horizon',h,'dates',len(a),'avgN %.2f IC %.6f ICIR %.6f hit %.4f'%(np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)))
 if h==1:
  for lo,hi in [(2020,2021),(2022,2023),(2024,2025),(2026,2027)]:
   q=a[[lo<=d.year<=hi for d in ds]]; print('regime',lo,hi,'n',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage %.4f turnover %.4f stress_share %.4f'%(F.notna().sum().sum()/F.size,F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),stress.mean()))
print('instruments',len(U))
