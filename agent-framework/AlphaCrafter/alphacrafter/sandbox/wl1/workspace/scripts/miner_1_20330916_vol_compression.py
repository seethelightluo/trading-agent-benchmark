import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f):
  d=pd.read_csv(f); P[a]=d.set_index(pd.to_datetime(d.date)).close
P=pd.DataFrame(P).sort_index().ffill(); ret=P.pct_change()
# volatility compression plus positive drift: low recent vol relative to long vol, gated by positive 20d return
compression=(1-ret.rolling(10).std()/(ret.rolling(60).std()+1e-8))*np.sign(P.pct_change(20))
sig=compression.shift(1); out=[]
for h in [5,10,20]:
 fwd=P.shift(-h)/P-1; z=[]
 for dt in sig.index:
  x=sig.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8:z.append(spearmanr(x[ok],y[ok]).statistic)
 z=pd.Series(z); print('h',h,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
fwd=P.shift(-10)/P-1
for lo,hi in [('2020','2021-12-31'),('2022','2024-12-31'),('2025','2027-12-31'),('2028','2030-12-31'),('2031','2033-12-31')]:
 z=[]
 for dt in sig.loc[lo:hi].index:
  x=sig.loc[dt];y=fwd.loc[dt];ok=x.notna()&y.notna()
  if ok.sum()>=8:z.append(spearmanr(x[ok],y[ok]).statistic)
 z=pd.Series(z); print(lo, len(z),z.mean(),z.mean()/z.std(ddof=1))
print('coverage',sig.notna().sum(axis=1).mean()/len(A),'turnover',sig.rank(pct=True).diff().abs().mean(axis=1).mean())
sig.to_csv('scripts/miner_1_20330916_vol_compression_signal.csv')
