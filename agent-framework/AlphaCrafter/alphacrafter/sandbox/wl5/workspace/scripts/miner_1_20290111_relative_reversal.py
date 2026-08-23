import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv(os.path.join('../persistent/stock_data',s+'.csv'),parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:'2029-01-10'].astype(float)
r=P.pct_change(); ret10=P/P.shift(10)-1; ret30=P/P.shift(30)-1
v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std()
# Relative reversal: buy assets that underperformed peers over 10d, with stronger signal in high cross-sectional dispersion, but avoid persistent trends.
rel=ret10.sub(ret10.median(axis=1),axis=0)
disp=ret10.std(axis=1); dscale=(disp/disp.rolling(60,min_periods=30).median()).clip(.5,2)
trend=(ret30.abs()/(v60*np.sqrt(30))).clip(0,1)
F=(-rel/v20).mul(dscale,axis=0)*(1-trend)
F.to_csv('scripts/miner_1_20290111_relative_reversal_signal.csv')
def calc(h,start=None,end=None):
 ic=[]; cov=[]; tr=[]
 for i in range(len(P)-h):
  d=str(P.index[i].date())
  if start and not(start<=d<=end): continue
  z=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   ic.append(spearmanr(z.f,z.y).statistic); cov.append(len(z)/15)
  if i>0:
   q=pd.concat([F.iloc[i-1].rank().rename('a'),F.iloc[i].rank().rename('b')],axis=1).dropna()
   if len(q)>=8: tr.append((q.a-q.b).abs().mean()/14)
 x=np.array(ic); return len(x),np.mean(cov),np.mean(x),np.mean(x)/np.std(x,ddof=1),np.mean(x>0),np.mean(tr)
print('range',P.index.min(),P.index.max(),'assets',len(U))
for h in [5,10,20]: print('horizon',h,calc(h=h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-01-01','2029-01-10')]: print('regime',a,b,calc(10,a,b))
