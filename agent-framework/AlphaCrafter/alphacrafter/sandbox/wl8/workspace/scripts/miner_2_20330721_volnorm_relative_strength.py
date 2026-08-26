import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2033-07-20'); p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill(); r=p.pct_change()
def ic(a,b):
 ok=a.notna()&b.notna()
 if ok.sum()<8:return np.nan
 return spearmanr(a[ok],b[ok]).statistic
# Volatility-normalized 20d relative strength; risk-adjusted trend, lagged one day
for L in [20,40]:
 vol=r.rolling(20).std()*np.sqrt(252); f=(p.shift(1)/p.shift(L+1)-1)/vol.shift(1)
 f=f.sub(f.median(axis=1),axis=0).rolling(3,min_periods=3).mean(); print('\nLOOK',L)
 for h in [1,5,10,20]:
  z=[];ns=[]
  for i,d in enumerate(p.index):
   if d<pd.Timestamp('2020-03-01') or d>cut or i+h>=len(p):continue
   q=ic(f.loc[d],(p.shift(-h)/p-1).loc[d])
   if pd.notna(q):z.append(q);ns.append((f.loc[d].notna()).sum())
  z=pd.Series(z);print(h,len(z),np.mean(ns),z.mean(),z.mean()/z.std(ddof=1),(z>0).mean())
 z=pd.Series(z);print('coverage',f.loc[(p.index>=pd.Timestamp('2020-03-01'))&(p.index<=cut)].notna().mean().mean(),'turnover',f.rank(pct=True).diff().abs().mean().mean())
