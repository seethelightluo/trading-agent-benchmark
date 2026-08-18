import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
TODAY=pd.Timestamp('2032-06-10')
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p=f'../persistent/stock_data/{s}.csv'
 if os.path.exists(p):
  d=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index(); D[s]=d.loc[d.index<=TODAY]
close=pd.DataFrame({s:d.close for s,d in D.items()}); high=pd.DataFrame({s:d.high for s,d in D.items()}); low=pd.DataFrame({s:d.low for s,d in D.items()})
hh=high.rolling(45,min_periods=30).max(); ll=low.rolling(45,min_periods=30).min(); f=.5-(close-ll)/(hh-ll).replace(0,np.nan)
def calc(ret):
 out=[]; ns=[]; dates=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],ret.loc[dt]],axis=1).dropna()
  if len(a)>=8: out.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic); ns.append(len(a)); dates.append(dt)
 return np.array(out),np.array(ns),dates
print('factor=range_location_reversal_45d cutoff',TODAY.date())
for h in [5,10,20]:
 z,ns,ds=calc(close.shift(-h)/close-1); ir=z.mean()/z.std(ddof=1)
 print(f'H{h} dates={len(z)} avgN={ns.mean():.2f} coverage={f.notna().sum().sum()/(f.shape[0]*len(U)):.4f} IC={z.mean():.8f} ICIR={ir:.8f} hit={np.mean(z>0):.4f}')
z,ns,ds=calc(close.shift(-10)/close-1)
for a,b in [('2020','2023'),('2024','2027'),('2028','2032')]:
 q=np.array([v for d,v in zip(ds,z) if a<=str(d.date())[:4]<=b])
 print(f'regime {a}-{b} dates={len(q)} IC={q.mean():.8f} ICIR={q.mean()/q.std(ddof=1):.8f} hit={np.mean(q>0):.4f}')
print(f'turnover={f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean():.6f} end={f.index.max().date()}')
