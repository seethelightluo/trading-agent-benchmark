import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24')
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(sym, path='../persistent/stock_data/'):
 d=pd.read_csv(path+sym+'.csv',parse_dates=['date']).sort_values('date').set_index('date')
 return d[d.index<=cut].close.pct_change()
R=pd.DataFrame({a:load(a) for a in assets})
# Observation-only macro input is allowed as a feature, never an order.
rate=load('US10Y')
# rolling asset-specific sensitivity to rate shocks, multiplied by recent shock;
# negative exposure is expected to benefit when rates reverse.
beta=R.rolling(60,min_periods=40).cov(rate).div(rate.rolling(60,min_periods=40).var(),axis=0)
shock=rate.rolling(3,min_periods=3).sum()
F=(-beta.mul(shock,axis=0)).clip(-10,10)
F.to_csv('scripts/miner_1_20270325_rate_beta_signal.csv')
fwd=R.shift(-1); vals=[]; ds=[]; ns=[]
for dt in F.index:
 z=pd.concat([F.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
  vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ds.append(dt); ns.append(len(z))
s=pd.Series(vals,index=ds)
print('asset-specific rate beta shock reversal; dates',len(s),'avgN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4),'coverage',round(F.notna().sum(axis=1).mean()/15,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
 q=s[(s.index>=lo)&(s.index<=hi)]; print(lo+'-'+hi,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)
for h in [5,10]:
 fw=R.shift(-h).rolling(h).sum() # end-to-end forward sum approximately
 v=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:v.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('horizon',h,'IC',round(np.mean(v),6),'n',len(v))
