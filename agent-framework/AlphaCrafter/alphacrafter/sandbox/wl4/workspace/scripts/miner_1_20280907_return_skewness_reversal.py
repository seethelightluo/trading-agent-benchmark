import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P='../persistent/stock_data'
pr=pd.DataFrame({s:pd.read_csv(os.path.join(P,s+'.csv'),parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index(); r=pr.pct_change()
# Negative recent return skew: assets with positively skewed recent returns tend to mean revert; lagged, interpretable.
f=(-r.rolling(30,min_periods=20).skew()).shift(1)
def obs(dt,h):
 if dt not in r.index:return np.nan,0
 fut=(1+r.loc[dt:].iloc[1:h+1]).prod()-1
 z=pd.concat([f.loc[dt],fut],axis=1).dropna()
 if len(z)<8 or z.iloc[:,0].nunique()<2:return np.nan,len(z)
 return spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)
for h in [1,5,10,20]:
 a=[];ns=[]
 for dt in pr.index:
  x,n=obs(dt,h)
  if np.isfinite(x):a.append(x);ns.append(n)
 a=pd.Series(a);print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round((a>0).mean(),4))
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2028-09-06')]:
 a=[]
 for dt in pr.loc[lo:hi].index:
  x,n=obs(dt,10)
  if np.isfinite(x):a.append(x)
 a=pd.Series(a);print('REG',lo,'dates',len(a),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5))
