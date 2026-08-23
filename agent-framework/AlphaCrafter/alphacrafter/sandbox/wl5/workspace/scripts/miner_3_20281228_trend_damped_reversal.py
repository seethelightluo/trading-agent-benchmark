import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv(os.path.join('../persistent/stock_data',s+'.csv'),parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:'2028-12-27'].astype(float)
r=P.pct_change(); ret5=P/P.shift(5)-1; vol20=r.rolling(20,min_periods=15).std()
disp=ret5.std(axis=1); med=disp.rolling(60,min_periods=30).median(); scale=(disp/med).clip(.5,2)
# Reversal is reduced when a persistent 60-day directional trend is strong.
ret60=P/P.shift(60)-1; vol60=r.rolling(60,min_periods=40).std()*np.sqrt(60)
trend_strength=(ret60.abs()/vol60).clip(0,1)
F=(-ret5/vol20).mul(scale,axis=0)*(1-trend_strength)
F.to_csv('scripts/miner_3_20281228_trend_damped_reversal_signal.csv')

def calc(h=10,start=None,end=None):
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
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-27'),('2028-01-01','2028-12-27')]: print('regime',a,b,calc(10,a,b))
for n in [63,126,252]:
 x=[]
 for i in range(len(P)-10):
  d=str(P.index[i].date())
  if d<'2028-01-01': continue
  z=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+10]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.f,z.y).statistic)
 print('recent observations',len(x),'mean',np.mean(x),'icir',np.mean(x)/np.std(x,ddof=1))
