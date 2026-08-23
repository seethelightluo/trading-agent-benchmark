import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv(os.path.join('../persistent/stock_data',s+'.csv'),parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:'2028-12-13'].astype(float)
r=P.pct_change(); vol=r.rolling(20,min_periods=15).std(); ret5=P/P.shift(5)-1
disp=ret5.std(axis=1); med=disp.rolling(60,min_periods=30).median()
# Continuous activation: relative dispersion, capped to limit outlier leverage.
scale=(disp/med).clip(0.5,2.0)
F=(-ret5/vol).mul(scale,axis=0)
print('dates',P.index.min(),P.index.max(),'assets',P.shape[1],'usable rows',F.notna().any(axis=1).sum())
def calc(start=None,end=None,h=10):
 vals=[]; cov=[]; turns=[]
 for i in range(len(P)-h):
  d=str(P.index[i].date())
  if start and not(start<=d<=end): continue
  z=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   vals.append(spearmanr(z.f,z.y).statistic); cov.append(len(z)/15)
  if i>0:
   zt=pd.concat([F.iloc[i-1].rank().rename('a'),F.iloc[i].rank().rename('b')],axis=1).dropna()
   if len(zt)>=8: turns.append((zt.a-zt.b).abs().mean()/14)
 x=np.array(vals); return len(x),np.mean(cov),np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0),np.mean(turns)
for h in [5,10,20]: print('horizon',h,calc(h=h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-13'),('2028-01-01','2028-12-13')]: print('regime',a,b,calc(a,b,10))
F.to_csv('scripts/miner_3_20281214_continuous_dispersion_reversal_signal.csv')
